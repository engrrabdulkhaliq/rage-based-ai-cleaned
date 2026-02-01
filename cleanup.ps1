Remove-Item -Path ".vscode" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".devcontainer" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".orchids" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "main.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "reorganize.ps1" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "*.pyc" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "temp_chunk_video*.mp3" -Force -ErrorAction SilentlyContinue
