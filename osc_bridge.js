const osc = require('osc');
const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Serve static files (HTML, JS, CSS)
app.use(express.static(__dirname));

// Serve index.html at root
app.get('/', function(req, res) {
    const indexPath = path.join(__dirname, 'index.html');
    console.log('Attempting to serve:', indexPath);
    
    if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
    } else {
        res.status(404).send('index.html not found at: ' + indexPath);
    }
});

app.get('/sample', function(req, res) {
    const samplePath = path.join(__dirname, '/samples/humming.mp3');
    
    if (fs.existsSync(samplePath)) {
        // Set proper headers for MP3 files
        res.set({
            'Content-Type': 'audio/mpeg',
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range'
        });
        
        // Handle range requests for audio streaming
        const stat = fs.statSync(samplePath);
        const fileSize = stat.size;
        const range = req.headers.range;
        
        if (range) {
            const parts = range.replace(/bytes=/, "").split("-");
            const start = parseInt(parts[0], 10);
            const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
            const chunksize = (end - start) + 1;
            
            res.status(206);
            res.set({
                'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                'Content-Length': chunksize.toString()
            });
            
            const stream = fs.createReadStream(samplePath, { start, end });
            stream.pipe(res);
        } else {
            res.set('Content-Length', fileSize.toString());
            const stream = fs.createReadStream(samplePath);
            stream.pipe(res);
        }
    } else {
        res.status(404).send('Sample not found at: ' + samplePath);
    }
});

app.get('/poem', function(req, res) {
    const samplePath = path.join(__dirname, '/samples/Gedicht.mp3');
    console.log('Sample requested:', samplePath);
    
    if (fs.existsSync(samplePath)) {
        // Set proper headers for MP3 files
        res.set({
            'Content-Type': 'audio/mpeg',
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range'
        });
        
        // Handle range requests for audio streaming
        const stat = fs.statSync(samplePath);
        const fileSize = stat.size;
        const range = req.headers.range;
        
        if (range) {
            const parts = range.replace(/bytes=/, "").split("-");
            const start = parseInt(parts[0], 10);
            const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
            const chunksize = (end - start) + 1;
            
            res.status(206);
            res.set({
                'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                'Content-Length': chunksize.toString()
            });
            
            const stream = fs.createReadStream(samplePath, { start, end });
            stream.pipe(res);
        } else {
            res.set('Content-Length', fileSize.toString());
            const stream = fs.createReadStream(samplePath);
            stream.pipe(res);
        }
    } else {
        res.status(404).send('Sample not found at: ' + samplePath);
    }
});

for (let i = 0; i < 4; i += 1) {
    app.get(`/p${i+1}`, function(req, res) {
        const samplePath = path.join(__dirname, `/samples/p${i+1}.mp3`);

        if (fs.existsSync(samplePath)) {
            // Set proper headers for MP3 files
            res.set({
                'Content-Type': 'audio/mpeg',
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                'Access-Control-Allow-Headers': 'Range'
            });
            
            // Handle range requests for audio streaming
            const stat = fs.statSync(samplePath);
            const fileSize = stat.size;
            const range = req.headers.range;
            
            if (range) {
                const parts = range.replace(/bytes=/, "").split("-");
                const start = parseInt(parts[0], 10);
                const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
                const chunksize = (end - start) + 1;
                
                res.status(206);
                res.set({
                    'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                    'Content-Length': chunksize.toString()
                });
                
                const stream = fs.createReadStream(samplePath, { start, end });
                stream.pipe(res);
            } else {
                res.set('Content-Length', fileSize.toString());
                const stream = fs.createReadStream(samplePath);
                stream.pipe(res);
            }
        } else {
            res.status(404).send('Sample not found at: ' + samplePath);
        }
    });
}

// Add OPTIONS handler for CORS preflight requests
app.options('/sample', (req, res) => {
    res.set({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range'
    });
    res.sendStatus(200);
});

app.options('/poem', (req, res) => {
    res.set({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range'
    });
    res.sendStatus(200);
});

for (let i = 0; i < 4; i += 1) {
    app.options(`/p${i+1}`, (req, res) => {
        res.set({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range'
        });
        res.sendStatus(200);
    });
}

// OSC UDP Port (receiving from Python)
const udpPort = new osc.UDPPort({
    localAddress: "127.0.0.1",
    localPort: 8000, 
    metadata: true
});

// OSC UDP Port (sending to Processing)
const processingPort = new osc.UDPPort({
    localAddress: "127.0.0.1",
    localPort: 0, // Auto-assign
    remoteAddress: "127.0.0.1",
    remotePort: 8001,
    metadata: true
});

udpPort.open();
processingPort.open();

udpPort.on("ready", function () {
    console.log("✅ OSC UDP port is ready and listening on port 8000");
});

udpPort.on("error", function (error) {
    console.error("❌ OSC UDP port error:", error);
});

udpPort.on("message", function (oscMsg) {
    console.log('📨 Received OSC message:', {
        address: oscMsg.address,
        args: oscMsg.args,
        timestamp: new Date().toISOString()
    });
    
    // Forward OSC messages to all connected WebSocket clients
    const values = oscMsg.args.map(arg => arg.value);
    console.log('🔄 Processing OSC:', oscMsg.address, values);
    
    // Special logging for people positions (new endpoint from skeleton2.py)
    if (oscMsg.address === '/people/positions') {
        console.log('👥 People positions received:', values);
        if (values.length === 0) {
            console.log('   (No people detected)');
        } else {
            const numPeople = values.length / 4; // Groups of 4: [id, x, y, hands_raised]
            console.log(`   (${numPeople} people detected)`);
            
            // Log each person's data
            for (let i = 0; i < values.length; i += 4) {
                if (i + 3 < values.length) {
                    const id = values[i];
                    const x = values[i + 1];
                    const y = values[i + 2];
                    const handsRaised = values[i + 3];
                    const handsStatus = handsRaised ? '🙌' : '👇';
                    console.log(`   Person ${id}: (${x.toFixed(3)}, ${y.toFixed(3)}) meters, hands: ${handsStatus}`);
                }
            }
        }
    }
    
    // Legacy logging for old visitor positions (keep for backwards compatibility)
    if (oscMsg.address === '/visitors') {
        console.log('👥 Visitor positions (legacy):', values);
    }
    
    // Emit to Socket.IO clients with the OSC address as the event name
    io.emit(oscMsg.address, values);
    console.log('🔗 Forwarded to WebSocket clients');
    
    // Forward to Processing
    processingPort.send(oscMsg);
});

io.on('connection', function(socket) {
    console.log('Client connected');
    socket.on('disconnect', function() {
        console.log('Client disconnected');
    });
});

http.listen(8080, function() {
    console.log('OSC bridge listening on port 8080');
    console.log('Receiving OSC on port 8000'); // This was already correct
    console.log('Open http://localhost:8080 in your browser');
});
