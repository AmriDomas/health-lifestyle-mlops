# Save as live_monitor.ps1
# Run with: .\live_monitor.ps1

while($true) {
    Clear-Host
    Write-Host "=== 🚀 MLOPs LIVE MONITORING DASHBOARD ===" -ForegroundColor Magenta
    Write-Host "Time: $(Get-Date)" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Health check
        $health = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 2
        Write-Host "✅ Backend Status: $($health.status)" -ForegroundColor Green
        Write-Host "   Models Loaded: $($health.models_loaded)" -ForegroundColor Cyan
        Write-Host "   Total Predictions: $($health.total_predictions)" -ForegroundColor Cyan
        
        # Prometheus metrics
        $predictions = Invoke-RestMethod "http://localhost:9090/api/v1/query?query=ml_predictions_total" -TimeoutSec 2
        $predictionCount = $predictions.data.result[0].value[1]
        
        $rate = Invoke-RestMethod "http://localhost:9090/api/v1/query?query=rate(ml_predictions_total[5m])" -TimeoutSec 2
        $predictionRate = $rate.data.result[0].value[1]
        
        $healthStatus = Invoke-RestMethod "http://localhost:9090/api/v1/query?query=up{job=`"health-ml-api`"}" -TimeoutSec 2
        $apiHealth = $healthStatus.data.result[0].value[1]
        
        Write-Host ""
        Write-Host "📊 Live Metrics:" -ForegroundColor White
        Write-Host "   Predictions: $predictionCount" -ForegroundColor Green
        Write-Host "   Rate: $predictionRate/sec" -ForegroundColor Green
        Write-Host "   API Health: $apiHealth" -ForegroundColor Green
        
    } catch {
        Write-Host "❌ Services unavailable or starting..." -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "🔄 Refreshing in 3 seconds..." -ForegroundColor Gray
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
    Start-Sleep -Seconds 3
}