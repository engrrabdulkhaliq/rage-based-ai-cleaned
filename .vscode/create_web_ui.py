import os

structure = {
    "web": {
        "templates": {
            "index.html": ""
        },
        "static": {
            "style.css": "",
            "chat.js": ""
        },
        "server.py": ""
    }
}

def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

create_structure(".", structure)

print("✅ Web UI folder structure created successfully")
