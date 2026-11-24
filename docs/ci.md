# CI and the merge queue

The [CI workflow](../.github/workflows/ci.yml) runs on pushes to `main`, copied PR
branches named `pull-request/<number>`, and `merge_group: checks_requested`.
Copied PRs and merge groups run lint, builds, GPU tests, and the summary job.
Pushes to `main` run only the build matrix to publish reusable sccache entries,
without packaging artifacts, GPU tests, or the summary job. Required merge-group
checks cover integration before the commit reaches `main`.

## Get a PR tested

1. Open a PR from your fork and mark it ready when it is ready for review.
   [copy-pr-bot](../.github/copy-pr-bot.yaml) automatically syncs eligible ready PRs;
   automatic syncing is disabled for drafts.
2. If the bot requests approval, a maintainer reviews the exact commit and comments
   `/ok to test <full-head-SHA>` on the PR. Follow the bot's instructions after each
   update; approval of an older SHA does not approve new code.
3. Check that the bot's `pull-request/<number>` branch matches the PR's current head.
   A successful run for an older commit does not establish that the update passes.
4. Inspect the `ci` run in the PR checks or the repository's Actions page.

To compare the two commits, replace `NUMBER` with the PR number:

```bash
gh pr view NUMBER --repo NVIDIA/cuCascade --json headRefOid --jq .headRefOid
gh api repos/NVIDIA/cuCascade/git/ref/heads/pull-request/NUMBER --jq .object.sha
```

Lint runs first for copied PRs and merge groups. Copied PRs build and test CUDA 12
with stable RAPIDS and CUDA 13 with nightly RAPIDS. Merge groups build and test all
four configurations; `main` builds all four to refresh the shared compiler cache.
For copied-PR and merge-group runs, the GPU test matrix starts after all builds succeed.
Tests use the single-GPU `linux-amd64-gpu-t4-latest-1` runners; green CI does not
establish multi-GPU coverage.
The required `check` job succeeds only when lint, build, and test jobs all succeed.

## Enter the merge queue

After the current commit passes `check` and satisfies the repository's review
requirements, a maintainer selects **Add to merge queue** on the PR. At least one
approving review is required. Code-owner review applies only to files with owners
defined in a `CODEOWNERS` file. These requirements remain in effect in the queue.

GitHub creates a temporary merge-group commit containing the PR and current `main`.
The workflow's `merge_group` trigger is configured to check that commit. Its required
`check` must pass before GitHub squash-merges the PR. Passing PR checks alone does not complete this
integration check. Do not push to the temporary `gh-readonly-queue/` branches.

Repository settings checked on 2026-11-16 were:

| Setting | Value |
| --- | --- |
| Merge method | Squash |
| Minimum / maximum PRs per merge group | 1 / 1 |
| Concurrent merge groups being checked | 1 |
| Required-check timeout | 60 minutes |
| Merge strategy | `ALLGREEN` |

The successful queue path was exercised by [#291](https://github.com/NVIDIA/cuCascade/pull/291).
Track rollout evidence in [#260](https://github.com/NVIDIA/cuCascade/issues/260).
Repository settings remain authoritative if these values change.

If GitHub removes a PR from the queue, inspect the failure, resolve it, and add the
PR again when eligible. Rerunning an old group does not put a removed PR back in the
queue. See GitHub's [merge-queue guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue).

## Diagnose and retry a failed run

Start with the failing lint, build, or test job. The summary `check` job reports their
outcome; rerunning only that summary cannot repair a failed dependency.

| Symptom | Next step |
| --- | --- |
| Formatting, compiler, or test assertion failure | Fix the change, push it, and check that the bot syncs the new SHA. |
| Queued job or unavailable runner | Check runner availability with a maintainer. A queued GPU job has not run tests. |
| Runner loss, network error, or artifact-service error | Inspect the logs and retry the failed jobs if the cause appears transient. Escalate repeated failures with the run URL and affected configuration. |
| Missing build artifact on a retry | Build artifacts expire after one day. Rerun the whole workflow to rebuild them. |
| Cancelled run for an older copied PR commit | Follow the run for the new head; superseded copied-PR runs are intentionally cancelled. |
| Failed or timed-out merge group | Inspect that group's run and queue status. Resolve the cause, then requeue the PR if it was removed. |

For a copied PR branch, retry only the run for its current head. Rerunning an older
commit shares the branch's concurrency group and can cancel the newer run.

Maintainers with permission to rerun jobs can use Actions or the CLI:

```bash
# Retry failed jobs and their dependents, including the summary job.
gh run rerun RUN_ID --repo NVIDIA/cuCascade --failed

# Rebuild and retest the original commit, including expired build artifacts.
gh run rerun RUN_ID --repo NVIDIA/cuCascade
```

Reruns use the original commit and ref. A source fix needs a new commit and a new
copied-PR run. Nightly dependencies can change between runs because `pixi.lock` is not
versioned. See GitHub's [rerun documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

When a test job uploads diagnostics, its job summary links the artifact containing
CTest JUnit results and `Testing/Temporary/` logs, retained for 14 days. Setup or
download failures and cancellation can leave only job logs. Save available
diagnostics before retrying if the failure needs investigation. CTest aggregates
test executables; inspect their output for hardware skips before claiming coverage.

Copied-PR cancellation applies only to superseded `pull-request/<number>` pushes.
`main` and merge-group runs are kept separate. Keep required checks and review
requirements in place while investigating infrastructure failures; involve a
maintainer when retrying does not resolve them.
