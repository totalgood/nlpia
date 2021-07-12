for p in *.adoc ; do
  echo "path: $p";
  pandoc -t json -o pandoc_json/"$p".json --indent=2 "$p"
done
