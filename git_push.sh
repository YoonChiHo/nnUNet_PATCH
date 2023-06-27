today=$(date "+%Y%m%d")

# Get private git code also
git config --global user.email "sciencedbs@gmail.com"
git config --global user.name "YoonChiHo"
git config --global user.password "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"

# Push
git add *
git commit -m ${today}'_update'
git pushhttps://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@github.com/YoonChiHo/nnUNet_PATCH.git
$ git push https://{token}@github.com/{username}/{repo_name}.git


echo PUSH ${today}_update Completed