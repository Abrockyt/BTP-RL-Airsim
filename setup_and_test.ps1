# AUTO-SETUP SCRIPT - Run this to fix everything automatically
# Usage: .\setup_and_test.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "AUTO-SETUP FOR DRONE SIMULATION" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Copy settings
Write-Host "[1/5] Installing AirSim settings..." -ForegroundColor Yellow
$settingsPath = "$env:USERPROFILE\Documents\AirSim"
if (-not (Test-Path $settingsPath)) {
    New-Item -Path $settingsPath -ItemType Directory -Force | Out-Null
}

Copy-Item ".\airsim_settings_dual_drone.json" "$settingsPath\settings.json" -Force

if (Test-Path "$settingsPath\settings.json") {
    Write-Host "✓ Settings installed" -ForegroundColor Green
    $droneCount = (Get-Content "$settingsPath\settings.json" | Select-String "Drone" | Measure-Object).Count
    Write-Host "  Found $droneCount drone configurations" -ForegroundColor Gray
} else {
    Write-Host "✗ Failed to copy settings" -ForegroundColor Red
    exit 1
}

# Step 2: Check Python
Write-Host "`n[2/5] Checking Python environment..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found" -ForegroundColor Red
    exit 1
}

# Step 3: Check AirSim
Write-Host "`n[3/5] Checking AirSim connection..." -ForegroundColor Yellow
Write-Host "  Running diagnostic..." -ForegroundColor Gray
$testResult = python test_drone_connection.py 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ AirSim connection successful" -ForegroundColor Green
    # Show summary
    $testResult | Select-String "Total drones detected" | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Gray
    }
} else {
    Write-Host "✗ AirSim connection failed" -ForegroundColor Red
    Write-Host "  Make sure AirSim/Unreal Engine is running!" -ForegroundColor Yellow
    Write-Host "`nError details:" -ForegroundColor Red
    Write-Host $testResult -ForegroundColor Red
    Write-Host "`nFix: Start AirSim, then run this script again" -ForegroundColor Yellow
    exit 1
}

# Step 4: Test movement
Write-Host "`n[4/5] Testing drone movement..." -ForegroundColor Yellow
Write-Host "  This will make Drone1 fly to (20, 20)..." -ForegroundColor Gray
Write-Host "  Watch the AirSim window - you should see movement!" -ForegroundColor Cyan

$moveResult = python test_movement.py 2>&1
if ($moveResult -match "GOAL REACHED") {
    Write-Host "✓ Movement test PASSED" -ForegroundColor Green
    Write-Host "  Drone successfully moved to target!" -ForegroundColor Gray
} else {
    Write-Host "⚠ Movement test unclear" -ForegroundColor Yellow
    Write-Host "  Check AirSim window - did the drone move?" -ForegroundColor Yellow
}

# Step 5: Verify latest code
Write-Host "`n[5/5] Verifying code version..." -ForegroundColor Yellow
$codeCheck = python -c "import smart_drone_vision_gui; print('MAIN_VEHICLE_NAME' in dir(smart_drone_vision_gui))" 2>&1
if ($codeCheck -match "True") {
    Write-Host "OK Latest code version confirmed" -ForegroundColor Green
} else {
    Write-Host "X Code version mismatch" -ForegroundColor Red
    Write-Host "  The smart_drone_vision_gui.py file may not be updated" -ForegroundColor Yellow
    exit 1
}

# Summary
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python smart_drone_vision_gui.py" -ForegroundColor White
Write-Host "  2. Click START FLIGHT button" -ForegroundColor White
Write-Host "  3. Drone should move within 10 seconds" -ForegroundColor White
Write-Host ""
Write-Host "Expected behavior:" -ForegroundColor Yellow
Write-Host "  - Position changes from (0,0) toward goal" -ForegroundColor Gray
Write-Host "  - Goal Distance decreases continuously" -ForegroundColor Gray
Write-Host "  - Drone visible moving in AirSim window" -ForegroundColor Gray
Write-Host "  - Setup completes in under 60 seconds" -ForegroundColor Gray
Write-Host ""
Write-Host "If issues persist, check:" -ForegroundColor Yellow
Write-Host "  - AirSim window is not minimized" -ForegroundColor Gray
Write-Host "  - Press 1 key to view Drone1" -ForegroundColor Gray
Write-Host "  - No other scripts controlling drones" -ForegroundColor Gray
Write-Host ""
Write-Host "Ready to start? (Y/N): " -NoNewline -ForegroundColor Cyan
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    Write-Host "`nLaunching GUI..." -ForegroundColor Green
    python smart_drone_vision_gui.py
} else {
    Write-Host "`nSetup complete. Run when ready." -ForegroundColor Gray
}
