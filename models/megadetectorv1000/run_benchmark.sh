#!/bin/bash
set -e

RESULTS_FILE="benchmark_results.json"
MDV5_PORT=8080
MDV1000_PORT=8080

echo "Starting MegaDetector benchmark comparison..."

# Clean up any existing results
rm -f "$RESULTS_FILE"

# Function to wait for endpoint to be ready
wait_for_endpoint() {
    local url=$1
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for endpoint to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "Endpoint ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "Endpoint failed to become ready"
    return 1
}

# Benchmark MDv5
echo ""
echo "=========================================="
echo "Benchmarking MDv5"
echo "=========================================="
cd ../megadetectorv5

docker run -d --name mdv5-benchmark -p $MDV5_PORT:8080 \
    -v $(pwd)/model_store:/opt/ml/model \
    torchserve-mdv5a:0.5.3-cpu

cd ../megadetectorv1000

wait_for_endpoint "http://localhost:$MDV5_PORT/ping"
sleep 5  # Extra warmup time

python3 benchmark.py "MDv5" "http://localhost:$MDV5_PORT/invocations" "$RESULTS_FILE"

docker stop mdv5-benchmark
docker rm mdv5-benchmark

# Benchmark MDv1000
echo ""
echo "=========================================="
echo "Benchmarking MDv1000"
echo "=========================================="

docker run -d --name mdv1000-benchmark -p $MDV1000_PORT:8080 megadetector-v1000

wait_for_endpoint "http://localhost:$MDV1000_PORT/ping"
sleep 5  # Extra warmup time

python3 benchmark.py "MDv1000" "http://localhost:$MDV1000_PORT/invocations" "$RESULTS_FILE"

docker stop mdv1000-benchmark
docker rm mdv1000-benchmark

# Print comparison
echo ""
echo "=========================================="
echo "COMPARISON RESULTS"
echo "=========================================="
python3 -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)

print(f\"{'Metric':<20} {'MDv5':<15} {'MDv1000':<15} {'Speedup':<10}\")
print('-' * 60)

mdv5 = next(r for r in results if r['model'] == 'MDv5')
mdv1000 = next(r for r in results if r['model'] == 'MDv1000')

print(f\"{'Mean':<20} {mdv5['mean']:<15.3f} {mdv1000['mean']:<15.3f} {mdv5['mean']/mdv1000['mean']:<10.2f}x\")
print(f\"{'Median':<20} {mdv5['median']:<15.3f} {mdv1000['median']:<15.3f} {mdv5['median']/mdv1000['median']:<10.2f}x\")
print(f\"{'P95':<20} {mdv5['p95']:<15.3f} {mdv1000['p95']:<15.3f} {mdv5['p95']/mdv1000['p95']:<10.2f}x\")
print(f\"{'Min':<20} {mdv5['min']:<15.3f} {mdv1000['min']:<15.3f}\")
print(f\"{'Max':<20} {mdv5['max']:<15.3f} {mdv1000['max']:<15.3f}\")

# Add comparison summary to JSON
comparison = {
    'mean_speedup': mdv5['mean'] / mdv1000['mean'],
    'median_speedup': mdv5['median'] / mdv1000['median'],
    'p95_speedup': mdv5['p95'] / mdv1000['p95']
}
results.append({'comparison': comparison})

with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
"

echo ""
echo "Full results saved to $RESULTS_FILE"
