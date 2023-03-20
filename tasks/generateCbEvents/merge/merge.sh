
for folder in $(cat folder.txt) ; do




    python haddnano.py out_$folder.root $1/$folder/*.root

    for f in $(cat to_remove.txt) ; do
        rm "$f"
    done

    rm ./to_remove.txt ./out_$folder.root

    python haddnano.py out_$folder.root $1/$folder/*.root



done
