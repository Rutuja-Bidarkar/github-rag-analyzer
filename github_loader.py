from github import Github

def fetch_repo_files(repo_url, github_token, max_files=150):
    repo_name = repo_url.rstrip("/").split("github.com/")[-1]
    g = Github(github_token)
    repo = g.get_repo(repo_name)

    allowed_ext = (".py", ".js", ".ts", ".java", ".md", ".json",
                    ".jsx", ".tsx", ".go", ".rb", ".cpp", ".c", ".html", ".css")

    excluded_files = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "composer.lock", "Gemfile.lock", "poetry.lock"
    }
    excluded_dirs = {
        "node_modules", ".git", "dist", "build", "__pycache__",
        ".next", "venv", "env", ".venv"
    }

    docs = []
    all_paths = []  # collects every file path for the tree summary

    def walk(path=""):
        contents = repo.get_contents(path)
        for item in contents:
            if item.type == "dir":
                if item.name in excluded_dirs:
                    continue
                walk(item.path)
            else:
                all_paths.append(item.path)  # track every file, even skipped ones
                if item.name in excluded_files:
                    continue
                if len(docs) >= max_files:
                    continue
                if item.name.endswith(allowed_ext):
                    try:
                        text = item.decoded_content.decode("utf-8", errors="ignore")
                        docs.append({"path": item.path, "content": text})
                    except Exception:
                        pass

    walk()

    # Add a synthetic document describing the full folder structure
    if all_paths:
        tree_text = "Repository file structure:\n" + "\n".join(sorted(all_paths))
        docs.insert(0, {"path": "__REPO_STRUCTURE__", "content": tree_text})

    return docs