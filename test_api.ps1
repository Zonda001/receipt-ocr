# Test script for Receipt OCR API
# Usage: .\test_api.ps1 -ImagePath "receipt.jpg" -Port 8000

param(
    [string]$ImagePath = "receipt.jpg",
    [int]$Port = 8000
)

$baseUrl = "http://localhost:$Port"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Receipt OCR API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test 1: Health check
Write-Host "`nTest 1: Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✓ API is healthy" -ForegroundColor Green
    Write-Host "  OCR Loaded: $($health.ocr_loaded)" -ForegroundColor Gray
    Write-Host "  Status: $($health.status)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 2: Root endpoint
Write-Host "`nTest 2: Root Endpoint" -ForegroundColor Yellow
try {
    $root = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "✓ Root endpoint working" -ForegroundColor Green
    Write-Host "  Version: $($root.version)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Scan receipt (if image provided)
if (Test-Path $ImagePath) {
    Write-Host "`nTest 3: Scanning Receipt" -ForegroundColor Yellow
    Write-Host "  Image: $ImagePath" -ForegroundColor Gray

    try {
        # Create form data
        $fileBin = [System.IO.File]::ReadAllBytes((Resolve-Path $ImagePath))
        $boundary = [System.Guid]::NewGuid().ToString()
        $LF = "`r`n"

        $bodyLines = (
            "--$boundary",
            "Content-Disposition: form-data; name=`"image`"; filename=`"$(Split-Path $ImagePath -Leaf)`"",
            "Content-Type: image/jpeg$LF",
            [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBin),
            "--$boundary--$LF"
        ) -join $LF

        $response = Invoke-RestMethod `
            -Uri "$baseUrl/scan/simple" `
            -Method Post `
            -ContentType "multipart/form-data; boundary=$boundary" `
            -Body $bodyLines

        Write-Host "✓ Receipt scanned successfully" -ForegroundColor Green
        Write-Host "  Texts found: $($response.texts.Count)" -ForegroundColor Gray

        if ($response.texts.Count -gt 0) {
            Write-Host "`n  First 10 texts:" -ForegroundColor Gray
            $response.texts[0..9] | ForEach-Object { Write-Host "    - $_" -ForegroundColor DarkGray }
        }

    } catch {
        Write-Host "✗ Scan failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "`nTest 3: Skipped (no image file: $ImagePath)" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan