@echo off
echo Adding files...
git add .

echo Committing changes...
git commit -m "%~1"

echo Pushing to origin...
git push origin master

echo Done! 