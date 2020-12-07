#!/bin/bash

HELP="`basename $0` [-h] [-i <argument>] [-r] [-f <argument>]"

#no arguments
if [ $# -eq 0 ]; then
    while read CUR_LINE; do
        echo "$CUR_LINE"
    done
  exit 0
fi

declare -a FILES
R_FLAG=false
I_ARG=""
F_ARG=""

while getopts ":hi:rf:" opt; do
  case $opt in
  i) 
    I_ARG="${OPTARG}";;
  r) 
    R_FLAG=true;;
  f) 
    F_ARG="${OPTARG}";;
  h) 
    echo $HELP; exit 0;;
  ?) 
    echo "Error: bad option: -${OPTARG}" >&2; exit 1;;
  esac
done
shift $(($OPTIND - 1))
if [ $# -gt 0 ]; then
  FILES=$@
fi

#read from standart output
if [ ${#FILES[*]} -eq 0 ]; then
	while read CUR_LINE; do
		#if I_ARG is empty, nothing will happen
		CUR_LINE=$(echo $CUR_LINE | sed -e 's|\(#[[:space:]]*include[[:space:]]*["]\)\([^"]*["]\)|\1'"$I_ARG"'\2|g')
		CUR_LINE=$(echo $CUR_LINE | sed -e 's|\(#[[:space:]]*include[[:space:]]*[<]\)\([^>]*[>]\)|\1'"$I_ARG"'\2|g')
		GREP_LINE=$(echo $CUR_LINE | grep -e "$F_ARG")
		if [ "$GREP_LINE" = "$CUR_LINE" ] && $R_FLAG; then
			CUR_LINE=$(echo "$CUR_LINE" | sed -e '/\(\<[a-z0-9]\+\(_[a-z0-9]\+\)\+[[:space:]]*(.*)\)/s/_\([a-z0-9]\)/\U\1/g')
		fi
		echo "$CUR_LINE"
	done
#read from files
else
	for FILE in ${FILES[@]}; do
		OLD_IFS=$IFS
		IFS=$'\n'
		cat "$FILE" | while read "CUR_LINE"; do
			CUR_LINE=$(echo $CUR_LINE | sed -e 's|\(#[[:space:]]*include[[:space:]]*["]\)\([^"]*["]\)|\1'"$I_ARG"'\2|g')
			CUR_LINE=$(echo $CUR_LINE | sed -e 's|\(#[[:space:]]*include[[:space:]]*[<]\)\([^>]*[>]\)|\1'"$I_ARG"'\2|g')
			GREP_LINE=$(echo $CUR_LINE | grep -e "$F_ARG")
				if [ "$GREP_LINE" = "$CUR_LINE" ] && $R_FLAG; then
					FUNCTIONS=$(echo $CUR_LINE | grep -o '\(\<[a-z0-9]\+\(_[a-z0-9]\+\)\+[[:space:]]*(\)')
					for FUNCTION in $FUNCTIONS; do
						NEW_FUNCTION=$(printf '%s' "$FUNCTION" | sed -e 's/_\([a-z0-9]\)/\U\1/g')
						CUR_LINE=$(echo "$CUR_LINE" | sed -e "s/$FUNCTION/$NEW_FUNCTION/g")
					done
				fi
			echo "$CUR_LINE" >> "$FILE"tmp
		done
		rm "$FILE"
		mv "$FILE"tmp "$FILE"
		
		IFS=$OLD_IFS
	done
fi

exit 0