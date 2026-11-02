# Issue tracker: GitHub

Issues and specs live in kevkrist/cuCascade.
Always pass --repo kevkrist/cuCascade to gh issue and label commands;
the clone's default gh repository points upstream.

## Operations

- Publish: gh issue create --repo kevkrist/cuCascade --title "<title>" --body-file <file> --label "<label>"
- Read: gh issue view <number> --repo kevkrist/cuCascade --json number,title,body,labels,comments
- List: gh issue list --repo kevkrist/cuCascade --state open --json number,title,body,labels
- Comment: gh issue comment <number> --repo kevkrist/cuCascade --body-file <file>
- Label: gh issue edit <number> --repo kevkrist/cuCascade --add-label "<label>"
- Close: gh issue close <number> --repo kevkrist/cuCascade

Use full URLs when referencing upstream issues or pull requests.

## Pull requests as a triage surface

PRs as a request surface: no.
