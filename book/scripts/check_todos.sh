#!/usr/bin/env bash

NUMTODOS=$(more manuscript/Chapter\ 03\ --\ Math\ with\ Words.asc | grep TODO | wc -l)

if [[ $NUMTODOS -gt 4 ]]; then
   echo $NUMTODOS
   exit $NUMTODOS
fi
echo "OK"
exit 0
