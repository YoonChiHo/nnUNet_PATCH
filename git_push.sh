
today=$(date "+%Y%m%d")

# Get private git code also
git config --global user.email "sciencedbs@gmail.com"
git config --global user.name "YoonChiHo"
git config --global user.password "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"

# Push
git add *
git commit -m ${today}'_update'
git push https://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@github.com/YoonChiHo/nnUNet_PATCH.git main # +강제

echo PUSH ${today}_update Completed

# Pull
#git pull https://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@github.com/YoonChiHo/nnUNet_PATCH.git main