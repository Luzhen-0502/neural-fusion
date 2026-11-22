# NF-1 架构检查器 · 一键安装
# 用法：irm https://github.com/Luzhen-0502/neural-fusion/releases/latest/download/install.ps1 | iex

$ErrorActionPreference = 'Stop'
$repo = 'Luzhen-0502/neural-fusion'
$exe  = 'NF1-Inspector.exe'
$url  = "https://github.com/$repo/releases/latest/download/$exe"
$dst  = Join-Path $env:USERPROFILE "Desktop\$exe"

Write-Host ''
Write-Host '  NF-1 架构检查器' -ForegroundColor Cyan
Write-Host '  ------------------------------' -ForegroundColor DarkGray
Write-Host '  说明：这是【架构查看器】，不含任何模型权重（约 10MB）。' -ForegroundColor DarkGray
Write-Host '        双击后能看 NF-1 每一层的张量维度和融合关系，' -ForegroundColor DarkGray
Write-Host '        还能一键跑形状验证（纯数学，不联网、不用 GPU）。' -ForegroundColor DarkGray
Write-Host ''

if (Test-Path $dst) {
    $sz = [math]::Round((Get-Item $dst).Length / 1MB, 2)
    Write-Host "  桌面已存在 $exe（$sz MB），直接覆盖..." -ForegroundColor Yellow
}

Write-Host '  正在下载...' -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing
} catch {
    Write-Host "  直连失败，改用镜像重试..." -ForegroundColor Yellow
    $mirror = "https://gh-proxy.com/$url"
    Invoke-WebRequest -Uri $mirror -OutFile $dst -UseBasicParsing
}

$info = Get-Item $dst
Write-Host ("  下载完成：{0:N2} MB" -f ($info.Length / 1MB)) -ForegroundColor Green

# 建快捷方式
$lnk = Join-Path $env:USERPROFILE 'Desktop\NF-1 架构检查器.lnk'
$w = New-Object -ComObject WScript.Shell
$s = $w.CreateShortcut($lnk)
$s.TargetPath = $dst
$s.WorkingDirectory = Split-Path $dst
$s.IconLocation = "$dst,0"
$s.Description = 'NF-1 架构检查器'
$s.Save()

Write-Host ''
Write-Host '  装好了！' -ForegroundColor Green
Write-Host "  桌面双击「NF-1 架构检查器」即可打开" -ForegroundColor Green
Write-Host ''
Write-Host '  想看源码/训练方案：https://github.com/' -NoNewline -ForegroundColor DarkGray
Write-Host $repo -ForegroundColor Cyan
Write-Host ''
