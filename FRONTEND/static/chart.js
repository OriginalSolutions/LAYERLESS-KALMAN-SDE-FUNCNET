// WERSJA: wspólny indeks rows, brak prawego odstępu,
// kółeczka na obu wykresach – indeks wymuszony bez szukania po x.

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded. Initializing HARD-SYNC Shared-Index Chart System.");

    const priceCanvas = document.getElementById('priceChart');
    const accuracyCanvas = document.getElementById('accuracyChart');
    if (!priceCanvas || !accuracyCanvas) {
        console.error("Canvas elements not found.");
        return;
    }

    const ctxPrice = priceCanvas.getContext('2d');
    const ctxAccuracy = accuracyCanvas.getContext('2d');

    let priceChart, accuracyChart;
    let isLiveMode = true;
    const liveModeButton = document.getElementById('live-mode-button');
    let localDataCache = null;   // { rows: [...] }
    let lastHoveredTimestamp = null;

    const EventBus = {
        _listeners: {},
        on(event, callback) {
            if (!this._listeners[event]) this._listeners[event] = [];
            this._listeners[event].push(callback);
        },
        emit(event, data) {
            if (this._listeners[event]) {
                this._listeners[event].forEach(cb => cb(data));
            }
        }
    };

    const verticalLinePlugin = {
        id: 'verticalLine',
        afterDraw: (chart) => {
            const active = chart.tooltip?.getActiveElements();
            if (!active || active.length === 0 || !active[0].element) return;
            const x = active[0].element.x;
            const { ctx, scales: { y } } = chart;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, y.top);
            ctx.lineTo(x, y.bottom);
            ctx.lineWidth = 1;
            ctx.strokeStyle = '#AAAAAA';
            ctx.stroke();
            ctx.restore();
        }
    };
    Chart.register(verticalLinePlugin);

    async function fetchData(start_ts = null, end_ts = null) {
        try {
            let url = '/history';
            if (start_ts && end_ts) url += `?start_ts=${start_ts}&end_ts=${end_ts}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Network error: ${resp.status}`);
            return await resp.json();
        } catch (e) {
            console.error("Data fetch failed:", e);
            return null;
        }
    }

    function processAndValidateData(apiResponse) {
        if (!apiResponse || !apiResponse.Time?.length) return null;

        const times   = apiResponse.Time;
        const price   = apiResponse.Actual_BTC_Price;
        const kama    = apiResponse.KAMA_Actual_BTC_Price;
        const fc      = apiResponse.Forecast;
        const arimax  = apiResponse.ARIMAX_Forecast;
        const kamaFc  = apiResponse.KAMA_Forecast;
        const accA    = apiResponse.Accuracy_ARIMAX;
        const accF    = apiResponse.Accuracy_Final;

        const len = times.length;
        const rows = [];

        for (let i = 0; i < len; i++) {
            const t = times[i];
            if (t == null) continue;
            rows.push({
                x: t,
                price:   Number.isFinite(price?.[i])  ? price[i]   : null,
                kama:    Number.isFinite(kama?.[i])   ? kama[i]    : null,
                fc:      Number.isFinite(fc?.[i])     ? fc[i]      : null,
                arimax:  Number.isFinite(arimax?.[i]) ? arimax[i]  : null,
                kamaFc:  Number.isFinite(kamaFc?.[i]) ? kamaFc[i]  : null,
                accA:    Number.isFinite(accA?.[i])   ? accA[i]    : null,
                accF:    Number.isFinite(accF?.[i])   ? accF[i]    : null,
            });
        }

        return { rows };
    }

    async function initialize() {
        const zoomPanOptions = {
            pan: {
                enabled: true,
                mode: 'xy',
                onPanStart: () => EventBus.emit('interaction:start'),
                onPan: ({ chart }) =>
                    EventBus.emit('zoompan:sync', {
                        sourceId: chart.canvas.id,
                        range: { min: chart.scales.x.min, max: chart.scales.x.max }
                    }),
            },
            zoom: {
                wheel: { enabled: true },
                pinch: { enabled: true },
                mode: 'xy',
                onZoomStart: () => EventBus.emit('interaction:start'),
                onZoom: ({ chart }) =>
                    EventBus.emit('zoompan:sync', {
                        sourceId: chart.canvas.id,
                        range: { min: chart.scales.x.min, max: chart.scales.x.max }
                    }),
                onZoomComplete: () => autoscaleAccuracyYAxis()
            }
        };

        const createChartOptions = (chartId) => ({
            responsive: true,
            maintainAspectRatio: false,
            parsing: false,
            animation: false,
            interaction: { mode: 'index', intersect: false, axis: 'x' },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#e0e0e0', usePointStyle: true, pointStyle: 'rect' },
                    position: 'top',
                    align: 'start'
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    usePointStyle: true,
                    callbacks: {
                        label: (context) => {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y != null) {
                                if (chartId === 'priceChart') {
                                    label += Math.round(context.parsed.y);
                                } else {
                                    label += context.parsed.y.toFixed(2) + '%';
                                }
                            }
                            return label;
                        }
                    }
                },
                zoom: zoomPanOptions
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'MM-dd HH:mm',
                            hour: 'MM-dd HH:mm',
                            day: 'yyyy-MM-dd'
                        }
                    },
                    ticks: {
                        display: chartId !== 'priceChart',
                        color: '#bcbcbc',
                        maxTicksLimit: 4,
                        minRotation: 0,
                        maxRotation: 0,
                        align: 'start'
                    },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: {
                        color: '#bcbcbc',
                        maxTicksLimit: chartId === 'accuracyChart' ? 5 : undefined,
                        callback: (value) => {
                            if (chartId === 'accuracyChart') return value.toFixed(2) + '%';
                            return Math.round(value);
                        }
                    }
                }
            }
        });

        priceChart = new Chart(ctxPrice, {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Actual Price',    data: [], borderColor: '#FFBF00', backgroundColor: '#FFBF00', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#FFBF00' },
                    { label: 'KAMA (Price)',    data: [], borderColor: '#FFFFFF', backgroundColor: '#FFFFFF', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#FFFFFF' },
                    { label: 'Forecast',        data: [], borderColor: '#4169E1', backgroundColor: '#4169E1', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#4169E1' },
                    { label: 'ARIMAX Forecast', data: [], borderColor: '#FF6347', backgroundColor: '#FF6347', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#FF6347', borderDash: [5, 5] },
                    { label: 'KAMA (Forecast)', data: [], borderColor: '#00FFFF', backgroundColor: '#00FFFF', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#00FFFF' }
                ]
            },
            options: createChartOptions('priceChart')
        });

        accuracyChart = new Chart(ctxAccuracy, {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Accuracy ARIMAX (%)', data: [], borderColor: '#9370DB', backgroundColor: '#9370DB', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#9370DB' },
                    { label: 'Accuracy Final (%)',  data: [], borderColor: '#32CD32', backgroundColor: '#32CD32', pointRadius: 0, pointHoverRadius: 7, pointHoverBackgroundColor: '#32CD32' }
                ]
            },
            options: createChartOptions('accuracyChart')
        });

        EventBus.on('interaction:start', () => {
            if (isLiveMode) {
                isLiveMode = false;
                liveModeButton.style.display = 'inline-block';
            }
        });

        EventBus.on('zoompan:sync', (data) => {
            const charts = [priceChart, accuracyChart];
            const target = charts.find(c => c.canvas.id !== data.sourceId);
            if (!target) return;
            target.options.scales.x.min = data.range.min;
            target.options.scales.x.max = data.range.max;
            target.update('none');
        });

        const handleMouseMoveSync = (e, sourceChart) => {
            const rect = sourceChart.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const ts = sourceChart.scales.x.getValueForPixel(x);
            if (ts) {
                lastHoveredTimestamp = ts;
                updateTooltipsAndLines();
            }
        };

        priceCanvas.addEventListener('mousemove', (e) => handleMouseMoveSync(e, priceChart));
        accuracyCanvas.addEventListener('mousemove', (e) => handleMouseMoveSync(e, accuracyChart));

        const rawData = await fetchData();
        if (rawData) {
            localDataCache = processAndValidateData(rawData);
            if (localDataCache) updateChartData(localDataCache, true);
        }

        setInterval(async () => {
            const liveData = await fetchData();
            if (liveData) {
                const newCache = processAndValidateData(liveData);
                if (newCache) {
                    localDataCache = newCache;
                    updateChartData(localDataCache, isLiveMode);
                }
            }
        }, 3000);
    }

    function updateChartData(cache, shouldFollowLive) {
        if (!cache || !cache.rows) return;
        const rows = cache.rows;

        const mkSeries = (selector) => rows.map(r => ({ x: r.x, y: selector(r) }));

        priceChart.data.datasets[0].data = mkSeries(r => r.price);
        priceChart.data.datasets[1].data = mkSeries(r => r.kama);
        priceChart.data.datasets[2].data = mkSeries(r => r.fc);
        priceChart.data.datasets[3].data = mkSeries(r => r.arimax);
        priceChart.data.datasets[4].data = mkSeries(r => r.kamaFc);

        accuracyChart.data.datasets[0].data = mkSeries(r => r.accA);
        accuracyChart.data.datasets[1].data = mkSeries(r => r.accF);

        if (shouldFollowLive) {
            const lastX = rows.length ? rows[rows.length - 1].x : Date.now();
            const min = lastX - 10 * 60 * 1000;
            const max = lastX;  // brak odstępu po prawej
            priceChart.options.scales.x.min = min;
            priceChart.options.scales.x.max = max;
            accuracyChart.options.scales.x.min = min;
            accuracyChart.options.scales.x.max = max;
        }

        priceChart.update('none');
        accuracyChart.update('none');
        autoscaleAccuracyYAxis();
    }

    liveModeButton.addEventListener('click', () => {
        isLiveMode = true;
        liveModeButton.style.display = 'none';
        fetchData().then(rawData => {
            if (!rawData) return;
            localDataCache = processAndValidateData(rawData);
            if (!localDataCache) return;
            priceChart.resetZoom('none');
            accuracyChart.resetZoom('none');
            updateChartData(localDataCache, true);
        });
    });

    function autoscaleAccuracyYAxis() {
        if (!accuracyChart || !accuracyChart.data?.datasets?.length) return;
        const { min: xMin, max: xMax } = accuracyChart.scales.x;
        const vals = [];
        accuracyChart.data.datasets.forEach(ds => {
            ds.data.forEach(p => {
                if (p.x >= xMin && p.x <= xMax && p.y != null && isFinite(p.y)) {
                    vals.push(p.y);
                }
            });
        });
        if (!vals.length) return;
        const minY = Math.min(...vals);
        const maxY = Math.max(...vals);
        const pad = (maxY - minY) * 0.1 || 1;
        accuracyChart.options.scales.y.min = minY - pad;
        accuracyChart.options.scales.y.max = maxY + pad;
        accuracyChart.update('none');
    }

    // 3. Tooltipy i kółeczka: wspólny refIndex w rows dla wszystkich datasetów na obu wykresach
    function updateTooltipsAndLines() {
        if (!priceChart || !accuracyChart) return;

        if (lastHoveredTimestamp == null || !localDataCache?.rows?.length) {
            if (priceChart.tooltip) priceChart.tooltip.setActiveElements([], { x: 0, y: 0 });
            if (accuracyChart.tooltip) accuracyChart.tooltip.setActiveElements([], { x: 0, y: 0 });
            priceChart.update('none');
            accuracyChart.update('none');
            return;
        }

        const rows = localDataCache.rows;

        // 3.1 znajdź indeks rows najbliższy lastHoveredTimestamp
        let refIndex = 0;
        let minDiff = Infinity;
        for (let i = 0; i < rows.length; i++) {
            const d = Math.abs(rows[i].x - lastHoveredTimestamp);
            if (d < minDiff) {
                minDiff = d;
                refIndex = i;
            }
        }
        const refX = rows[refIndex].x;

        const setTooltipForChart = (chart, chartName) => {
            if (!chart?.data?.datasets?.length) return;

            const pixelX = chart.scales.x.getPixelForValue(refX);
            const active = [];

            chart.data.datasets.forEach((ds, di) => {
                // zakładamy, że data[refIndex] istnieje (lub jest y = null)
                if (ds.data[refIndex]) {
                    active.push({ datasetIndex: di, index: refIndex });
                }
            });

            console.log(`[ACTIVE] ${chartName} refIndex=${refIndex} refX=${new Date(refX).toISOString()} activeCount=${active.length}`);

            chart.tooltip?.setActiveElements(active, { x: pixelX, y: 0 });
            chart.update('none');
        };

        setTooltipForChart(priceChart, 'priceChart');
        setTooltipForChart(accuracyChart, 'accuracyChart');
    }

    initialize();
});
