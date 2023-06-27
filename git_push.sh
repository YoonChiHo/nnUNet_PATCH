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