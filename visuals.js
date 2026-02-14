// Visitor visual configuration
const VISITOR_RADIUS = 40;           // Increased from 40
const VISITOR_OUTLINE_THICKNESS = 3; // Configurable outline thickness
const VISITOR_OUTLINE_OPACITY = 30;  // Outline opacity

// Particle visual configuration
const PARTICLE_SIZE = 2;           // Particle stroke weight (was 0.9)

let visitors = [];
let visitorIdMap = new Map(); // Track visitors by ID

// Attractor State
let xn = 0.1, yn = 0.1;
let a = 1.5, b = -1.8, c = 1.2, d = 0.9;
let targetD = 0.9; // Target value for d parameter
let dSmoothingSpeed = 0.02; // How fast d transitions 
let t = 0;
let dt = 0.000001;

let pointsGeometry;

// Column notes
const resonanceSliceFrequencies = [73.42, 87.31, 110, 130.81, 82.41, 98];
let resonanceSlices = [];

const densityThreshold = ITERATIONS_PER_FRAME * 0.05;

// TRACKING
let lastSideMatrix;
let screenCentroidX = 0, screenCentroidY = 0;

// Manual visitor for testing
let manualVisitor = null;
const MANUAL_VISITOR_ID = 'manual-test-visitor';

// Entropy tracking
const entropyGridResolution = 30;
let   entropyGridOccupancy  = [];
let   entropyOccupiedCount  = 0;
let   currentEntropy        = 0;
let   lastEntropyState      = 0;

// Audio
let audioInitialized = false;

let hummingWavesSampler, hummingArpeggioSampler, poemSampler;
let proximityMelody, proximityPad;
let bass;

async function initAudio() {
  try {
    console.log("🎵 Starting audio initialization...");
    
    // Add timeout to prevent hanging
    const initTimeout = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Audio initialization timeout')), 10000)
    );
    
    const initPromise = (async () => {
      // Check if Tone is properly started
      if (Tone.context.state !== 'running') {
        console.warn("⚠️ Tone context not running, attempting to resume...");
        await Tone.context.resume();
        
        // Wait a bit for context to stabilize
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (Tone.context.state !== 'running') {
          throw new Error(`Tone context state is ${Tone.context.state}, expected 'running'`);
        }
      }
      
      console.log("Setting Transport BPM...");
      Tone.Transport.bpm.value = 120;

      console.log("Initializing synths...");
      // Test synth creation first
      const testSynth = new Tone.Oscillator(440, "sine");
      testSynth.dispose();
      console.log("✅ Synth creation test passed");
      
      // Initialize Synths ONLY after user gesture
      const darkPad = new DarkAmbientPad(); 
      resonanceSlices = SCALE_NOTES.map(note => new ResonanceSlice(note, darkPad));

      console.log("Loading samples...");
      hummingWavesSampler    = new AbstractHummingSampler("http://localhost:8080/sample");  // Use full URL
      hummingArpeggioSampler = new ArpeggioHumSampler("http://localhost:8080/sample");     // Use full URL
      poemSampler            = new PoemSampler();

      console.log("Setting up proximity systems...");
      proximityPad = new ProximityPad();
      for (let i = 0; i < visitors.length; i += 1) {
        visitors[i].volumeGate = new Tone.Gain(0.2);
        proximityPad.createVoice(SCALE_NOTES[i], visitors[i].volumeGate);
      }

      proximityMelody = new ProximityMelody();
      proximityMelody.start();

      console.log("Initializing bass synth...");
      bass = new BassSynth();
      
      console.log("Starting Transport...");
      Tone.Transport.start();
      
      // Verify transport is running
      await new Promise(resolve => setTimeout(resolve, 100));
      console.log("Transport state:", Tone.Transport.state);
    })();
    
    // Race between init and timeout
    await Promise.race([initPromise, initTimeout]);

    audioInitialized = true;
    console.log("✅ Audio initialization complete!");
    
  } catch (error) {
    console.error("❌ Audio initialization failed:", error);
    
    // Add more diagnostic info
    console.log("🔍 Audio Context Details:");
    console.log("- State:", Tone.context?.state);
    console.log("- Sample Rate:", Tone.context?.sampleRate);
    console.log("- Current Time:", Tone.context?.currentTime);
    console.log("- Destination:", Tone.context?.destination);
    
    audioInitialized = false;
    throw error; // Re-throw so the caller knows it failed
  }
}

// Socket.IO connection for OSC messages
let socket;

function setup() {
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent("visuals");
  background(0);
  pixelDensity(2);
  frameRate(60);
  
  // Connect to OSC bridge server
  console.log('Attempting to connect to Socket.IO server...');
  socket = io('http://localhost:8080');
  
  socket.on('connect', () => {
    console.log('✅ Connected to OSC bridge');
  });
  
  socket.on('connect_error', (error) => {
    console.error('❌ Connection error:', error);
  });
  
  // Listen for people positions from skeleton2.py
  // Expected format: [id1, x1, y1, id2, x2, y2, ...]
  socket.on('/people/positions', (args) => {
    // console.log('👥 People positions received:', args);
    updateVisitors(args);
  });
  
  // Listen for ANY OSC message to debug
  socket.onAny((eventName, ...args) => {
    // console.log('🔔 OSC event:', eventName, args);
  });
  
  socket.on('disconnect', () => {
    console.log('❌ Disconnected from OSC bridge');
  });

  // Start with no visitors - they will be added via OSC
  // (or keep default visitors as fallback)
  // visitors.push(new Visitor(width * 0.25, height * 0.25, 261.63, 'default-1'));
  // visitors.push(new Visitor(width * 0.15, height * 0.50, 329.63, 'default-2'));
  // visitors.push(new Visitor(width * 0.50, height * 0.75, 392.00, 'default-3'));
  // visitors.push(new Visitor(width * 0.50, height * 0.25, 392.00, 'default-4'));
  
  // Map default visitors
  visitors.forEach(v => visitorIdMap.set(v.id, v));

  // Calculate all point positions once per frame
  pointsGeometry = new p5.Geometry();
  
  // Add 10,000 vertices (these will be instanced)
  for (let i = 0; i < ITERATIONS_PER_FRAME; i++) {
    pointsGeometry.vertices.push(createVector(0, 0, 0)); // placeholder
  }
  
  lastSideMatrix = Array(visitors.length).fill(null).map(() => Array(visitors.length).fill(false));
  
  // Initialize entropy grid
  for (let i = 0; i < entropyGridResolution * entropyGridResolution; i++) {
    entropyGridOccupancy[i] = false;
  }
}

// Update visitors based on OSC data
// Expected format: [id1, x1, y1, hands1, id2, x2, y2, hands2, ...]
function updateVisitors(args) {
  if (!audioInitialized) return;

  if (!args || args.length === 0) {
    console.log('📍 No people detected - keeping existing visitors');
    // When no people detected, slowly return to base value
    targetD = 0.9;
    return;
  }
  
  console.log('📍 Updating visitors with data:', args);
  
  const newVisitorIds = new Set();
  const frequencies = [261.63, 329.63, 392.00, 440.00, 493.88, 523.25]; // C, E, G, A, B, C
  
  // Track if anyone has hands raised for global attractor parameter
  let anyHandsRaised = false;
  
  // Process visitor data in groups of 4 (id, x, y, hands_raised)
  for (let i = 0; i < args.length; i += 4) {
    if (i + 3 >= args.length) break;
    
    const id = String(args[i]);
    const x = args[i + 1];  // x coordinate in meters
    const y = args[i + 2];  // y coordinate in meters
    const handsRaised = Boolean(args[i + 3]); // hands raised status
    
    if (handsRaised) {
      anyHandsRaised = true;
    }
    
    newVisitorIds.add(id);
    
    // Map floor coordinates to screen space
    // Floor coordinate system: (0,0) = bottom-left, (3,3.5) = top-right
    const screenX = map(x, 0, 3, 0, width);
    const screenY = map(y, 0, 3.5, height, 0);  // Flip Y coordinate: larger Y values go toward top
    
    const handsStatus = handsRaised ? '🙌' : '👇';
    console.log(`📍 Person ${id}: floor(${x.toFixed(2)}, ${y.toFixed(2)}) -> screen(${screenX.toFixed(1)}, ${screenY.toFixed(1)}) hands: ${handsStatus}`);
    
    if (visitorIdMap.has(id)) {
      // Update existing visitor with smoothed position and hands data
      const visitor = visitorIdMap.get(id);
      const oldX = visitor.targetPos.x;
      const oldY = visitor.targetPos.y;
      visitor.setTargetPosition(screenX, screenY);
      visitor.handsRaised = handsRaised; // Update hands status
      
      const distance = dist(oldX, oldY, screenX, screenY);
      if (distance > 10) {
        console.log(`🎯 Updated visitor ${id} target: (${oldX.toFixed(1)}, ${oldY.toFixed(1)}) -> (${screenX.toFixed(1)}, ${screenY.toFixed(1)}) hands: ${handsStatus} [distance: ${distance.toFixed(1)}px]`);
      }
    } else {
      // Create new visitor
      const freqIndex = visitors.length % frequencies.length;
      const newVisitor = new Visitor(screenX, screenY, frequencies[freqIndex], id);
      newVisitor.handsRaised = handsRaised; // Set initial hands status
      newVisitor.volumeGate = new Tone.Gain(0.2);
      proximityPad.createVoice(SCALE_NOTES[freqIndex], newVisitor.volumeGate);

      visitors.push(newVisitor);
      visitorIdMap.set(id, newVisitor);
      console.log(`➕ Added visitor ${id} at floor coords (${x.toFixed(2)}, ${y.toFixed(2)}) -> screen (${screenX.toFixed(1)}, ${screenY.toFixed(1)}) hands: ${handsStatus}`);
    }
  }
  
  // Update target attractor parameter based on hands status (smoothly)
  targetD = anyHandsRaised ? 2.0 : 0.9;
  
  // Only log when target changes
  const prevTarget = d;
  if (Math.abs(targetD - prevTarget) > 0.01) {
    console.log(`🎯 Attractor target d set to: ${targetD} (hands raised: ${anyHandsRaised})`);
  }
  
  // Remove visitors that are no longer present
  const idsToRemove = [];
  visitorIdMap.forEach((visitor, id) => {
    if (!newVisitorIds.has(id) && !id.startsWith('default-')) {
      idsToRemove.push(id);
    }
  });
  
  idsToRemove.forEach(id => {
    const visitor = visitorIdMap.get(id);
    const index = visitors.indexOf(visitor);
    if (index > -1) {
      visitors.splice(index, 1);
      console.log(`➖ Removed visitor ${id}`);
    }
    visitorIdMap.delete(id);
  });
  
  // Update lastSideMatrix size if visitor count changed
  if (lastSideMatrix.length !== visitors.length) {
    lastSideMatrix = Array(visitors.length).fill(null).map(() => Array(visitors.length).fill(false));
  }
}

function resetSystem() {
  for (let v  of visitors)         v.resetHeat();
  for (let rs of resonanceSlices) rs.resetHeat();
  
  // Reset entropy grid
  for (let i = 0; i < entropyGridOccupancy.length; i++) { entropyGridOccupancy[i] = false; }
  
}
function draw() {
  if (!audioInitialized) {
    background(0);
    return;
  }

  // WebGL puts (0, 0) at the center of the sketch 
  // Therefore, we move origin to top-left
  // translate(-width/2, -height/2); 
  background(0);
  
  resetSystem();

  // Update visitor positions with smoothing
  for (let visitor of visitors) {
    visitor.updatePosition();
  }

  t += dt;

  // Smoothly interpolate d toward its target value
  d = lerp(d, targetD, dSmoothingSpeed);
  
  // Optional: Log when d changes significantly (for debugging)
  if (Math.abs(d - targetD) > 0.01) {
    // Still transitioning
    if (frameCount % 60 === 0) { // Log once per second
      console.log(`🔄 Attractor d transitioning: ${d.toFixed(3)} → ${targetD.toFixed(3)}`);
    }
  }

  strokeWeight(1);
  
  entropyOccupiedCount = 0;
  let minX = 999, maxX = -999, minY = 999, maxY = -999;
  let sumX = 0, sumY = 0;
  
  let breathe = Math.sin(frameCount * 0.005);
  let scalar = 1.0;

  beginShape(POINTS);
    strokeWeight(0.9);
  
    if (lastEntropyState === -1) {
      stroke(100, 100, 255, 140);
    } else {
      stroke(255, 140);
    }

    for (let i = 0; i < ITERATIONS_PER_FRAME; i++) {
      // Calculate next Clifford step
      let xn_1 = scalar * (
        Math.sin(a * yn) + c * Math.cos(a * xn) + Math.sin(t * width)
      );
    
      let yn_1 = scalar * (
        Math.sin(b * xn) + d * Math.cos(b * yn) + Math.cos(t * height)
      );
    
      // Map to pixels
      let px = map(xn_1, -4, 4, 0, width);
      let py = map(yn_1, -4, 4, 0, height);
      // Draw Point directly (no object creation)
      vertex(px, py);
    
      // Map to entropy grid
      let gx = Math.floor(map(px, 0, width, 0, entropyGridResolution - 1));
      let gy = Math.floor(map(py, 0, height, 0, entropyGridResolution - 1));
    
      if (gx >= 0 && gx < entropyGridResolution && gy >= 0 && gy < entropyGridResolution) {
        let index = gx + gy * entropyGridResolution;
      
        if (!entropyGridOccupancy[index]) {
          entropyGridOccupancy[index] = true;
          entropyOccupiedCount++;
        }
      }
    
      // Visitor interactions
      for (let v of visitors) {
        let pdx = v.pos.x - px;
        let pdy = v.pos.y - py;
        let pdistSq = pdx * pdx + pdy * pdy;
      
        if (pdistSq < v.radius * v.radius) v.registerHit();
      
        // Gravity coupling
        let vax = map(v.pos.x, 0, width, -3, 3);
        let vay = map(v.pos.y, 0, height, -3, 3);
        let dx = vax - xn_1;
        let dy = vay - yn_1;
        let dSq = dx * dx + dy * dy + SOFTENING;
      
        xn_1 += (dx / dSq) * G_STRENGTH;
        yn_1 += (dy / dSq) * G_STRENGTH;
      }
    
      xn = xn_1;
      yn = yn_1;

      if (xn < minX) minX = xn;
      if (xn > maxX) maxX = xn;
      if (yn < minY) minY = yn;
      if (yn > maxY) maxY = yn;
    
      sumX += xn;
      sumY += yn;

      // Column density
      let columnIndex = constrain(Math.floor(px / (width / resonanceSlices.length)), 0, resonanceSlices.length - 1);
      resonanceSlices[columnIndex].registerHit();
    }
  endShape();
  
  screenCentroidX = map(sumX / ITERATIONS_PER_FRAME, -4, 4, 0, width);
  screenCentroidY = map(sumY / ITERATIONS_PER_FRAME, -4, 4, 0, height);
  
  // === INTERACTIONS ===
  
  // 1. EXPANSION/CONTRACTION
  currentEntropy = entropyOccupiedCount / (entropyGridResolution * entropyGridResolution);
  
  if (currentEntropy < 0.05 && lastEntropyState !== -1) {
    bass.playNote(SCALE_NOTES[0], "1n", Tone.now());
    lastEntropyState = -1;
  } else if (currentEntropy > 0.15 && lastEntropyState !== 1) {
    lastEntropyState = 1;
    bass.playNote(SCALE_NOTES[2], "1n", Tone.now());
  }

  hummingWavesSampler.updateWaves(currentEntropy, frameCount);
  hummingArpeggioSampler.update(currentEntropy);
  poemSampler.update(screenCentroidX);
  
  for (let i = 0; i < visitors.length; i++) {
    let v1 = visitors[i];

    for (let j = i + 1; j < visitors.length; j++) {
      if (i == j) continue;
      
      let v2 = visitors[j];
      
      // Check between-visitor proximity 
      let d = dist(v1.pos.x, v1.pos.y, v2.pos.x, v2.pos.y);

      if (d < 200) {
        let midX = (v1.pos.x + v2.pos.x) / 2;
        let midY = (v1.pos.y + v2.pos.y) / 2;
        let radius = v1.radius + v2.radius + d/4;
  
        proximityMelody.update(i, j, d);
      }
      
      // Centroid has crossed a line connecting two visitors
      let sideVal = (screenCentroidX - v1.pos.x) * (v2.pos.y - v1.pos.y) -
                    (screenCentroidY - v1.pos.y) * (v2.pos.x - v1.pos.x);

      let currentSide = sideVal > 0;
      
      if (currentSide !== lastSideMatrix[i][j]) {
        poemSampler.trigger();
      }

      lastSideMatrix[i][j] = currentSide;
    }
  }
  
  // Update all synths
  for (let v of visitors) {
    v.display();
    v.updateSound();
  }
  
  for (let i = 0; i < resonanceSlices.length; i++) {
    resonanceSlices[i].updateSound(i);
  }
}

class Visitor {
  constructor(x, y, freq, id = null) {
    this.pos = createVector(x, y);
    this.targetPos = createVector(x, y);  // Target position from OSC
    this.freq = freq;
    this.hitCount = 0;
    this.radius = VISITOR_RADIUS;  // Use configurable radius
    this.synth = null;
    this.id = id || `visitor-${Math.random().toString(36).substr(2, 9)}`;
    this.handsRaised = false; // New property for hands status
    
    // Smoothing parameters - made more responsive
    this.smoothing = 0.25;  // Increased from 0.15 for more responsiveness
    this.deadZone = 2;      // Reduced from 5 for more sensitivity
  }
  
  // Update target position (called from OSC updates)
  setTargetPosition(x, y) {
    this.targetPos.x = x;
    this.targetPos.y = y;
  }
  
  // Smooth position updates (called every frame)
  updatePosition() {
    // Calculate distance to target
    let distance = dist(this.pos.x, this.pos.y, this.targetPos.x, this.targetPos.y);
    
    // Only move if outside dead zone
    if (distance > this.deadZone) {
      // Lerp towards target position
      this.pos.x = lerp(this.pos.x, this.targetPos.x, this.smoothing);
      this.pos.y = lerp(this.pos.y, this.targetPos.y, this.smoothing);
    }
  }
  
  setSynth(synth) {
    this.synth = synth;
  }
  
  resetHeat() {
    this.hitCount = 0;
  }
  
  registerHit() {
    this.hitCount++;
  }
  
  updateSound() {
    let targetVolume = 0;
    
    // Same logic as ResonanceSlice
    if (this.hitCount > 5) {
      targetVolume = map(
        this.hitCount, 
        0, 
        ITERATIONS_PER_FRAME * 0.5,
        0.05, 
        0.25
      );

      targetVolume = constrain(targetVolume, 0.05, 0.25);
      targetVolume *= MASTER_VOLUME;
    }
    
    // Smoothly ramp volume (no trigger/release)
    this.volumeGate.gain.rampTo(targetVolume, 0.2);
  }
  
  display() {
    // Update position before displaying
    this.updatePosition();
    
    noFill();
    
    // Use different colors based on hands status
    if (this.handsRaised) {
      stroke(255, 255, 0, 80); // Yellow when hands are raised
    } else {
      stroke(255, 40);         // White when hands are down
    }
    ellipse(this.pos.x, this.pos.y, this.radius * 2);
    
    if (this.hitCount > 5) {
      let visualCappedHits = constrain(this.hitCount, 0, DENSITY_THRESHOLD);
      let opacity   = map(visualCappedHits, 0, DENSITY_THRESHOLD, 0, 150);
      let pulseSize = map(visualCappedHits, 0, DENSITY_THRESHOLD, this.radius * 2, 120);
      
      // Different pulse colors based on hands status
      if (this.handsRaised) {
        fill(255, 255, 0, opacity); // Yellow pulse when hands raised
      } else {
        fill(255, opacity);         // White pulse when hands down
      }
      noStroke();
      circle(this.pos.x, this.pos.y, pulseSize);
    }
  }
}

class ResonanceSlice {
  constructor(note, pad) {
    this.note = note;
    this.hitCount = 0;
    this.pad = pad;
    
    // Create volume gate FIRST
    this.volumeGate = new Tone.Gain(0); // Start silent
    
    // Tell the pad to create a voice for this note with this volume gate
    this.pad.createVoice(note, this.volumeGate);
  }
  
  resetHeat() {
    this.hitCount = 0;
  }
  
  registerHit() {
    this.hitCount++;
  }
  
  updateSound(columnIndex) {
    let targetVolume = 0;
    
    if (this.hitCount > DENSITY_THRESHOLD) {
      targetVolume = map(
        this.hitCount, 
        DENSITY_THRESHOLD, 
        ITERATIONS_PER_FRAME * 0.5,
        0, 
        0.15
      );
      targetVolume = Math.pow(targetVolume, 2) * 6; 
      targetVolume = constrain(targetVolume, 0, 0.015);
      targetVolume *= MASTER_VOLUME;
    }
    
    // Smoothly ramp volume (no trigger/release)
    this.volumeGate.gain.rampTo(targetVolume, 0.2);

    // Visual feedback
    if (this.hitCount > DENSITY_THRESHOLD) {
      let alpha = map(targetVolume, 0, 0.15, 0, 10);
      alpha = constrain(alpha, 0, 1);
      
      fill(255, 255, 255, alpha);
      noStroke();
      let colWidth = width / resonanceSlices.length;
      rect(columnIndex * colWidth, 0, colWidth, height);
    }
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

// Mouse click handler to toggle manual visitor
function mousePressed() {
  if (manualVisitor) {
    // Remove existing manual visitor
    const index = visitors.indexOf(manualVisitor);
    if (index > -1) {
      visitors.splice(index, 1);
      console.log(`➖ Removed manual visitor at (${manualVisitor.pos.x.toFixed(1)}, ${manualVisitor.pos.y.toFixed(1)})`);
    }
    visitorIdMap.delete(MANUAL_VISITOR_ID);
    manualVisitor = null;
    
    // Update lastSideMatrix size
    lastSideMatrix = Array(visitors.length).fill(null).map(() => Array(visitors.length).fill(false));
  } else {
    // Add new manual visitor at mouse position
    const frequencies = [261.63, 329.63, 392.00, 440.00, 493.88, 523.25];
    const freqIndex = visitors.length % frequencies.length;
    
    manualVisitor = new Visitor(mouseX, mouseY, frequencies[freqIndex], MANUAL_VISITOR_ID);
    
    if (audioInitialized) {
      manualVisitor.volumeGate = new Tone.Gain(0.2);
      proximityPad.createVoice(SCALE_NOTES[freqIndex], manualVisitor.volumeGate);
    }
    
    visitors.push(manualVisitor);
    visitorIdMap.set(MANUAL_VISITOR_ID, manualVisitor);
    console.log(`➕ Added manual visitor at (${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`);
    
    // Update lastSideMatrix size
    lastSideMatrix = Array(visitors.length).fill(null).map(() => Array(visitors.length).fill(false));
  }
}

// Update mouse position for manual visitor
function mouseMoved() {
  if (manualVisitor) {
    manualVisitor.setTargetPosition(mouseX, mouseY);
  }
}
