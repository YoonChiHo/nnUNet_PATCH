today=$(date "+%Y%m%d")

# Push
git add *
git commit -m ${today}'_update'
git push https://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@github.com/YoonChiHo/nnUNet_PATCH.git


echo PUSH ${today}_update Completed