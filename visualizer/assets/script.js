var eventSource = new EventSource('sse');

let eventQueue = [];
let batchSize = 10;  // default
let processedCount = 0;
let isPaused = false;

const ignoredEvents = [
    'config:load', 
    'index:built', 
    'model:train_finished', 
    'model:train_started', 
    'clustering:compute_labels',
    'index:buffer_flush',
];

eventSource.onmessage = (event) => {
    const eventData = JSON.parse(JSON.parse(event.data));
    if (ignoredEvents.includes(event['message'])) {
        // skip
        return;
    }
    eventQueue.push(eventData);
    processQueue();
}

const processQueue = () => {
    if (isPaused) return;

    while (processedCount < batchSize && eventQueue.length > 0) {
        const event = eventQueue.shift();
        evalEvent(event);
        processedCount++;
    }
    renderBucketsByLevel();
    $('#queue-length').empty();
    $('#queue-length').append(`Event Queue Length: <span id="queue-length-value">${eventQueue.length}</span>`);

    if (processedCount >= batchSize) {
        isPaused = true;
    }
}

$('#process-btn').on('click', () => {
    const userValue = parseInt($('#event-limit').val(), 10);
    if (!isNaN(userValue) && userValue > 0) {
        batchSize = userValue;
    }
    processedCount = 0;
    isPaused = false;
    processQueue();
});


$('#process-all-btn').on('click', () => {
    batchSize = eventQueue.length;
    processedCount = 0;
    isPaused = false;
    processQueue();
});

const evalEvent = (event) => {
    console.log('Message from server', typeof event);
    switch (event['message']) {
        case 'bucket:insert':
            bucketInsert(event); break;
        case 'time':
            timeEvent(event); break;
        case 'index:cluster_shape':
            indexClusterShape(event); break;
        case 'bucket:insert_many':
            bucketInsertMany(event); break;
        case 'bucket:rescale':
            bucketRescale(event); break;
        case 'metrics':
            metrics(event); break;
        default:
            console.warn("Unknown event", event['message']);
    }
};

const bucketsByLevel = {};  // {level: {bucket_idx: Bucket}}

const getBucketHtml = (bucket) => {
    return `
        <div class="bucket bg-white p-2 rounded shadow mr-4 mb-2" style="width: 150px;">
            <div class="text-sm font-medium text-gray-600 mb-1">Bucket: ${bucket.id}</div>
            <div class="w-full bg-gray-200 h-6 rounded">
                <div class="${bucket.occupied / bucket.size > 1.0 ? 'bg-red-600' : 'bg-blue-500'} h-6 rounded" style="width: ${(Math.min(bucket.occupied / bucket.size, 1.0)) * 100}%;"></div>
            </div>
            <div class="text-xs text-gray-500 mt-1">Occupied: ${bucket.occupied} / ${bucket.size}</div>
            <div class="text-xs text-gray-500 mt-1">Rescaled: ${bucket.rescaled}</div>
        </div>
    `;
}

const renderBucketsByLevel = () =>  {
    $('#bucket-view').empty();
    
    const buffer = bucketsByLevel['buffer'];
    const levelRow = $('<div>').addClass('level-row mb-4');
    levelRow.append(`<div class="text-lg font-bold mb-2">Level Buffer</div>`);
    const levelRows = $('<div>').addClass('flex');
    levelRows.append(getBucketHtml(buffer));
    $('#bucket-view').append(levelRows);

    Object.keys(bucketsByLevel).filter(level => level !== 'buffer').forEach(level => {
        const levelBuckets = bucketsByLevel[level];
        
        const levelRow = $('<div>').addClass('level-row mb-4');
        levelRow.append(`<div class="text-lg font-bold mb-2">Level ${level}</div>`);
        const phase = trainingPhases[level];
        levelRow.append(`<div class="space-y-2"><div class="text-gray-600 text-sm">Cluster Shape: ${phase.clusterShape}</div></div>`)
        const levelRows = $('<div>').addClass('flex');;
        Object.keys(levelBuckets).forEach(bucketIdx => {
            if (!(bucketIdx in levelBuckets)) {
                const bucketHtml = `
                    <div class="bucket bg-white p-2 rounded shadow mr-4 mb-2" style="width: 150px;">
                        Not used
                    </div>
                `;
                levelRow.append(bucketHtml);        
            } else {
                const bucket = levelBuckets[bucketIdx];
                const bucketHtml = getBucketHtml(bucket);
                levelRows.append(bucketHtml);        
            }
        });
        levelRow.append(levelRows);
        $('#bucket-view').append(levelRow);
    });
}


const bucketInsert = (event) => {
    const id = event.id;
    const occupied = event.occupied + 1;
    const size = event.size;

    const [_level, bucketIndex] = id.split(":");
    const level = _level === undefined ? 'buffer' : _level;
    if (!(level in bucketsByLevel)) {
        bucketsByLevel[level] = {};
    }
    let rescaled = 0;
    if (bucketIndex in bucketsByLevel[level]) {
        rescaled = bucketsByLevel[level][bucketIndex].rescaled;
    }

    const bucket = {id, occupied, size, rescaled};
    if (level === 'buffer') {
        bucketsByLevel[level] = bucket;
    } else {
        bucketsByLevel[level][bucketIndex] = bucket;
    }
};


const bucketInsertMany = (event) => {
    const id = event.id;
    const size = event.size;
    const occupied = event.occupied + event.values_len;

    const [_level, bucketIndex] = id.split(":");
    const level = _level === undefined ? 'buffer' : _level;
    
    if (!(level in bucketsByLevel)) {
        bucketsByLevel[level] = [];
    }
    let rescaled = 0;
    if (bucketIndex in bucketsByLevel[level]) {
        rescaled = bucketsByLevel[level][bucketIndex].rescaled;
    }

    const bucket = {id, occupied, size, rescaled};
    if (level === 'buffer') {
        bucketsByLevel[level] = bucket;
    } else {
        bucketsByLevel[level][bucketIndex] = bucket;
    }
};

const bucketRescale = (event) => {
    const id = event.id;
    const [level, bucketIndex] = id.split(":");
    const bucket = bucketsByLevel[level][bucketIndex];
    const rescaled = bucket.rescaled + 1;
    bucketsByLevel[level][bucketIndex] = {...bucket, rescaled};
};

const functionTimings = {}; // key: function name, value: array of ms times

function renderFunctionTimings() {
    $('#timing-list').empty();

    Object.entries(functionTimings).forEach(([fn, times]) => {
        const avg = (times.reduce((a, b) => a + b, 0)) / times.length;

        const timingDiv = $(`
            <div class="bg-white rounded p-2 shadow flex justify-between">
                <span class="font-medium">${fn}</span>
                <span class="text-gray-600">${avg.toFixed(3)} ms (n=${times.length})</span>
            </div>
        `);

        $('#timing-list').append(timingDiv);
    });
}

const timeEvent = (event) => {
    const fn = event.function;
    const timeStr = event.time;

    const timeMs = parseFloat(timeStr.replace("ms", ""));

    if (!functionTimings[fn]) {
        functionTimings[fn] = [];
    }

    functionTimings[fn].push(timeMs);

    renderFunctionTimings();
};

const trainingPhases = {};


const clearLevels = () => {
    const buffer = bucketsByLevel['buffer'];
    const occupied = 0;
    const rescaled = 0;
    bucketsByLevel['buffer'] = {...buffer, occupied, rescaled};
    Object.keys(bucketsByLevel).filter(level => level !== 'buffer').forEach(level => {
        const levelBuckets = bucketsByLevel[level];
        Object.keys(levelBuckets).forEach(bucketIdx => {
            const bucket = levelBuckets[bucketIdx];
            bucketsByLevel[level][bucketIdx] = {...bucket, occupied, rescaled};
        });
    });
}

const indexClusterShape = (event) => {
    const levelIdx = event.level_idx;
    const clusterShape = event.cluster_shape;
    trainingPhases[levelIdx] = {clusterShape, levelIdx};
    clearLevels();
};

const metrics = (event) => {
    const r1 = event.recall_top1 || 0;
    const r5 = event.recall_top5 || 0;
    const r10 = event.recall_top10 || 0;

    const scale = x => `${Math.min(x * 100, 100)}%`;

    $('#recall1').css('width', scale(r1));
    $('#recall1').append(r1);
    $('#recall5').css('width', scale(r5));
    $('#recall5').append(r5);
    $('#recall10').css('width', scale(r10));
    $('#recall10').append(r10);
};

const modelTrainStarted = (event) => {

};

const modelTrainFinished = (event) => {

};
