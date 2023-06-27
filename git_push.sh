
today=$(date "+%Y%m%d")

# Get private git code also
git config --global user.email "sciencedbs@gmail.com"
git config --global user.name "YoonChiHo"
git config --global user.password "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"

# Push
git add *
git commit -m ${today}'_update'
git push https://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@github.com/YoonChiHo/nnUNet_PATCH.git !main

echo PUSH ${today}_update Completed

# Pull
#git pull https://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@github.com/YoonChiHo/nnUNet_PATCH.git main