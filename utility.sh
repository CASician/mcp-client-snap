# utility.sh

function center_text() {
    # La logica di centratura che ti ho dato prima
    local COLS=${COLUMNS:-$(tput cols)} 
    local TEXT="$1"
    local INDENT=$(( (COLS - ${#TEXT}) / 2 ))
    printf "%*s%s\n" "$INDENT" "" "$TEXT"
}

# Per poter eseguire la funzione come un comando, devi aggiungerne un'altra
# che la invochi:
center_text "$@" 
# Il "$@" assicura che tutti gli argomenti passati allo script 
# vengano passati alla funzione center_text.
