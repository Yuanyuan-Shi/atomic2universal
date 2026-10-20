# Source this file, then call `load_dotenv <file>`: exports every KEY=VALUE line of <file> (spaces around
# '=' and surrounding quotes allowed, '#' comments and blank lines skipped) without echoing any value.
# `source <file>` cannot parse `KEY = value` lines, which the project's .env uses.
load_dotenv() {
  local file="$1" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ ! "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      echo "ERROR: $file has a line that is not KEY=VALUE (value not shown)" >&2; return 1
    fi
    key="${BASH_REMATCH[2]}" value="${BASH_REMATCH[3]}"
    value="${value%"${value##*[![:space:]]}"}"  # trailing whitespace
    if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
    export "$key=$value"
  done < "$file"
}
