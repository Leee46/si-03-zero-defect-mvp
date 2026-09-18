async function detect() {
    const formData = new FormData();
    formData.append('image', document.getElementById('fileInput').files[0]);

    const response = await fetch('/detect', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Detected: ${result.class} (Confidence: ${result.confidence.toFixed(2)})`;

    // Simulate causal analysis
    const causes = await getCausalAnalysis(result.class);
    document.getElementById('causes').innerText = `Top Causes: ${causes.join(', ')}`;
}

async function getCausalAnalysis(defectClass) {
    const causes = [
        "High Machine Temp",
        "Operator Shift Change",
        "Humidity Fluctuation"
    ];
    return causes.slice(0, 3);
}
