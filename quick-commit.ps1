# Quick Git Commit Script for PowerShell
# Usage: .\quick-commit.ps1 "Your commit message"

param(
    [Parameter(Mandatory=$true)]
    [string]$Message
)

Write-Host "🚀 Starting git workflow..." -ForegroundColor Green

# Add all changes
Write-Host "📁 Adding files..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ([string]::IsNullOrEmpty($status)) {
    Write-Host "✅ No changes to commit" -ForegroundColor Green
    exit 0
}

# Commit changes
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m $Message

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit successful" -ForegroundColor Green
    
    # Push to origin
    Write-Host "🌐 Pushing to origin..." -ForegroundColor Yellow
    git push origin master
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 Successfully pushed to Git!" -ForegroundColor Green
    } else {
        Write-Host "❌ Push failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Commit failed" -ForegroundColor Red
    exit 1
} 