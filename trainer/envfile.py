"""DRfC 의 run.env / system.env 읽기·쓰기. 원본의 주석과 줄 순서를 최대한 보존한다."""
import re

_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*?)\s*$")


def parse_env(text):
    """주석이 아닌 KEY=VALUE 줄만 읽는다. 값에 들어 있는 $VAR 참조는 풀지 않고 그대로 둔다."""
    out = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            v = m.group(2)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[m.group(1)] = v
    return out


def fmt_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def set_env(text, updates):
    """updates: {KEY: 값 또는 None}. None 이면 해당 줄을 주석 처리한다.
    이미 있는 줄은 그 자리에서 바꾸고, 주석 처리된 같은 키가 있으면 살려서 바꾸고, 없으면 끝에 추가한다."""
    lines = text.splitlines()
    for key, val in updates.items():
        active = re.compile(rf"^\s*{re.escape(key)}=")
        commented = re.compile(rf"^\s*#\s*{re.escape(key)}=")
        idx = next((i for i, l in enumerate(lines) if active.match(l)), None)
        if idx is None:
            idx = next((i for i, l in enumerate(lines) if commented.match(l)), None)
        if val is None:
            if idx is not None and active.match(lines[idx]):
                lines[idx] = "# " + lines[idx].lstrip()
            continue
        new = f"{key}={fmt_value(val)}"
        if idx is None:
            lines.append(new)
        else:
            lines[idx] = new
    return "\n".join(lines) + "\n"
