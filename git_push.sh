today=$(date "+%Y%m%d")

# Get private git code also
git config --global user.email "sciencedbs@gmail.com"
git config --global user.name "YoonChiHo"
git config --global user.password "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"

# Push
git add *
git commit -m ${today}'_update'
git push https://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@github.com/YoonChiHo/nnUNet_PATCH.git


echo PUSH ${today}_update Completed