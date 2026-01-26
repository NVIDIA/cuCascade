import os
import xml.etree.ElementTree as ET
import sys


def parse_xml(xml_dir, output_file):
    index_path = os.path.join(xml_dir, "index.xml")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return

    tree = ET.parse(index_path)
    root = tree.getroot()

    classes = []
    for compound in root.findall("compound"):
        if compound.get("kind") in ["class", "struct"]:
            refid = compound.get("refid")
            name = compound.find("name").text
            classes.append({"name": name, "refid": refid})

    with open(output_file, "w") as f:
        f.write("# API Reference\n\n")
        f.write("## Class Hierarchy\n\n")
        f.write("```mermaid\nclassDiagram\n")

        all_relationships = set()
        class_details = []

        for cls in classes:
            cls_xml_path = os.path.join(xml_dir, f"{cls['refid']}.xml")
            if not os.path.exists(cls_xml_path):
                continue

            cls_tree = ET.parse(cls_xml_path)
            cls_root = cls_tree.getroot()
            compounddef = cls_root.find("compounddef")

            # Inheritance
            for base in compounddef.findall("basecompoundref"):
                base_name = base.text
                all_relationships.add(f"    {base_name} <|-- {cls['name']}")

            # Details
            brief = compounddef.find("briefdescription/para")
            brief_text = brief.text if brief is not None else ""

            methods = []
            for section in compounddef.findall("sectiondef"):
                if section.get("kind") == "public-func":
                    for member in section.findall("memberdef"):
                        method_name = member.find("name").text
                        m_brief = member.find("briefdescription/para")
                        m_brief_text = m_brief.text if m_brief is not None else ""
                        methods.append({"name": method_name, "brief": m_brief_text})

            class_details.append(
                {"name": cls["name"], "brief": brief_text, "methods": methods}
            )

        for rel in all_relationships:
            f.write(f"{rel}\n")
        f.write("```\n\n")

        f.write("## Classes\n\n")
        for cls in class_details:
            f.write(f"### {cls['name']}\n\n")
            if cls["brief"]:
                f.write(f"{cls['brief']}\n\n")

            if cls["methods"]:
                f.write("| Method | Description |\n")
                f.write("| --- | --- |\n")
                for m in cls["methods"]:
                    f.write(f"| `{m['name']}` | {m['brief']} |\n")
                f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_api_docs.py <xml_dir> <output_file>")
        sys.exit(1)
    parse_xml(sys.argv[1], sys.argv[2])
