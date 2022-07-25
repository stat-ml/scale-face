import sys

for line in sys.stdin:
    if "config." in line:
        parts = line.strip().split(" ")
        parts.remove("=")
        key = parts[0].replace("config.", "")
        value = " ".join(parts[1:])
        value = value.replace("True", "true").replace("False", "false")
        value = value.replace("None", "null")

        new_line = key + ": " + value + "\n"
    else:
        new_line = line
    sys.stdout.write(new_line)