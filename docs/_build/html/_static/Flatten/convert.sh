ls *-01.eps | cut -d "." -f1 > lst

cat lst | while read -r line; do

    ps2pdf -dEPSCrop ${line}.eps

    pdf2svg ${line}.pdf ${line}.svg

done

