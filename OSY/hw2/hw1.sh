#!/bin/bash

HELP="This script requests the path to the file and determines its type.
After starting enter the data as follows: PATH /example/.../...
The script have two optional arguments:
-h ... help
-z ... archive all regular files
"
RETURN_VALUE=0
Z_FLAG=false
declare -a FILES

#check parameters
for ARG in $@; do
   case "$ARG" in
   "-h")
      echo "$HELP"; exit 0;;
   "-z")
      Z_FLAG=true;;
   *)
      echo "Error: bad arguments" >&2; exit 2;;
   esac
done

#read dat
while read CUR_LINE; do
   if  [ "${CUR_LINE:0:5}" == "PATH " ]; then
      FILE=${CUR_LINE#"PATH "}
      if [ -L "$FILE" ]; then
         REAL_FILE=$(readlink "$FILE")
         echo "LINK '${FILE}' '${REAL_FILE}'"
      elif [ -d "$FILE" ]; then
         echo "DIR '${FILE}'"
      elif [ -f "$FILE" ]; then
         echo "FILE '${FILE}' $(wc -l < "$FILE" | xargs) '$(head -n 1 "$FILE")'"
         FILES+=("$FILE")
      else
         echo "ERROR '${FILE}'" >&2
         RETURN_VALUE=1
      fi
   fi
done

#archive
if $Z_FLAG; then
   tar czf output.tgz "${FILES[@]}"
fi

exit $RETURN_VALUE
