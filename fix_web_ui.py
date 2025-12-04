
import os

file_path = r"d:\project\ai\VideoKnowlege\web_ui.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Remove the last few lines if they match the problematic pattern
# We look for "if __name__" and replace it
cleaned_lines = []
found_main_guard = False
for line in lines:
    if 'if __name__ == "__main__":' in line:
        found_main_guard = True
        break
    cleaned_lines.append(line)

# Add the main guard back correctly
final_content = "".join(cleaned_lines).rstrip() + "\n\nif __name__ == \"__main__\":\n    main()\n"

with open(file_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Fixed web_ui.py")
