zip -r submission.zip \
models/ src/ \
docker-compose.yml Dockerfile \
SUBMISSION_README.md requirements.txt \
-x "*.gitignore" "*__pycache__*" "*Win_30_Stride_10_2022_04_27_18_45_11-e01val_accuracy0.9798.hdf5"