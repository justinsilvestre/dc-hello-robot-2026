function initializeP5Sketch(instruments, state) {

function setup() {
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent("visuals");
  background(0);
  pixelDensity(2);
  frameRate(60);




  // Calculate all point positions once per frame
  state.pointsGeometry = new p5.Geometry();
  
  // Add 10,000 vertices (these will be instanced)
  for (let i = 0; i < ITERATIONS_PER_FRAME; i++) {
    state.pointsGeometry.vertices.push(createVector(0, 0, 0)); // placeholder
  }
  
  state.lastSideMatrix = Array(state.visitors.length).fill(null).map(() => Array(state.visitors.length).fill(false));
  
  // Initialize entropy grid
  for (let i = 0; i < state.entropyGridResolution * state.entropyGridResolution; i++) {
    state.entropyGridOccupancy[i] = false;
  }
}

function resetSystem() {
  for (let v  of state.visitors)         v.resetHeat();
  for (let rs of state.resonanceSlices) rs.resetHeat();
  
  // Reset entropy grid
  for (let i = 0; i < state.entropyGridOccupancy.length; i++) { state.entropyGridOccupancy[i] = false; }
  
}
function draw() {
  const { visitors, resonanceSlices } = state;

  console.log("Draw call")
  // WebGL puts (0, 0) at the center of the sketch 
  // Therefore, we move origin to top-left
  // translate(-width/2, -height/2); 
  background(0);
  
  resetSystem();

  // Update visitor positions with smoothing
  for (let visitor of visitors) {
    visitor.updatePosition();
  }

  state.t += state.dt;

  // Smoothly interpolate d toward its target value
  state.d = lerp(state.d, state.targetD, state.dSmoothingSpeed);
  
  // Optional: Log when d changes significantly (for debugging)
  if (Math.abs(state.d - state.targetD) > 0.01) {
    // Still transitioning
    if (frameCount % 60 === 0) { // Log once per second
      console.log(`🔄 Attractor d transitioning: ${state.d.toFixed(3)} → ${state.targetD.toFixed(3)}`);
    }
  }

  strokeWeight(1);
  
  state.entropyOccupiedCount = 0;
  let minX = 999, maxX = -999, minY = 999, maxY = -999;
  let sumX = 0, sumY = 0;
  
  let breathe = Math.sin(frameCount * 0.005);
  let scalar = 1.0;

  beginShape(POINTS);
    strokeWeight(3);
  
    if (state.lastEntropyState === -1) {
      stroke(100, 100, 255, 140);
    } else {
      stroke(255, 140);
    }

    for (let i = 0; i < ITERATIONS_PER_FRAME; i++) {
      const { xn, yn, a, b, c, d, t } = state;
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
      let gx = Math.floor(map(px, 0, width, 0, state.entropyGridResolution - 1));
      let gy = Math.floor(map(py, 0, height, 0, state.entropyGridResolution - 1));
    
      if (gx >= 0 && gx < state.entropyGridResolution && gy >= 0 && gy < state.entropyGridResolution) {
        let index = gx + gy * state.entropyGridResolution;
      
        if (!state.entropyGridOccupancy[index]) {
          state.entropyGridOccupancy[index] = true;
          state.entropyOccupiedCount++;
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
    
      state.xn = xn_1;
      state.yn = yn_1;

      if (state.xn < minX) minX = state.xn;
      if (state.xn > maxX) maxX = state.xn;
      if (state.yn < minY) minY = state.yn;
      if (state.yn > maxY) maxY = state.yn;
    
      sumX += state.xn;
      sumY += state.yn;

      // Column density
      let columnIndex = constrain(Math.floor(px / (width / resonanceSlices.length)), 0, resonanceSlices.length - 1);
      resonanceSlices[columnIndex].registerHit();
    }
  endShape();
  
  state.screenCentroidX = map(sumX / ITERATIONS_PER_FRAME, -4, 4, 0, width);
  state.screenCentroidY = map(sumY / ITERATIONS_PER_FRAME, -4, 4, 0, height);
  
  // === INTERACTIONS ===
  
  // 1. EXPANSION/CONTRACTION
  state.currentEntropy = state.entropyOccupiedCount / (state.entropyGridResolution * state.entropyGridResolution);
  
  if (state.currentEntropy < 0.05 && state.lastEntropyState !== -1) {
    instruments.bass.playNote(SCALE_NOTES[0], "1n", Tone.now());
    state.lastEntropyState = -1;
  } else if (state.currentEntropy > 0.15 && state.lastEntropyState !== 1) {
    state.lastEntropyState = 1;
    instruments.bass.playNote(SCALE_NOTES[2], "1n", Tone.now());
  }

  instruments.hummingWavesSampler.updateWaves(state.currentEntropy, frameCount);
  instruments.hummingArpeggioSampler.update(state.currentEntropy);
  instruments.poemSampler.update(state.screenCentroidX);
  
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
  
        console.log(`Playing proximity melody for visitors ${v1.id} and ${v2.id} - distance: ${d.toFixed(1)}px`);
        instruments.proximityMelody.update(i, j, d);
      }
      
      // Centroid has crossed a line connecting two visitors
      let sideVal = (state.screenCentroidX - v1.pos.x) * (v2.pos.y - v1.pos.y) -
                    (state.screenCentroidY - v1.pos.y) * (v2.pos.x - v1.pos.x);

      let currentSide = sideVal > 0;
      
      if (currentSide !== state.lastSideMatrix[i][j]) {
        console.log(`Playing poem sampler for visitor pair (${v1.id}, ${v2.id}) - crossed line: ${currentSide ? 'left' : 'right'}`);
        instruments.poemSampler.trigger();
      }

      state.lastSideMatrix[i][j] = currentSide;
    }
  }
  
  // Update all synths
  for (let v of visitors) {
    v.display();
    v.updateSound();
  }
  
  for (let i = 0; i < resonanceSlices.length; i++) {
    resonanceSlices[i].updateSound(i, resonanceSlices);
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

// Mouse click handler to toggle manual visitor
function mousePressed() {
  if (state.manualVisitor) {
    // Remove existing manual visitor
    const index = visitors.indexOf(state.manualVisitor);
    if (index > -1) {
      visitors.splice(index, 1);
      console.log(`➖ Removed manual visitor at (${state.manualVisitor.pos.x.toFixed(1)}, ${state.manualVisitor.pos.y.toFixed(1)})`);
    }
    visitorIdMap.delete(MANUAL_VISITOR_ID);
    state.manualVisitor = null;
    
    // Update lastSideMatrix size
    state.lastSideMatrix = Array(visitors.length).fill(null).map(() => Array(visitors.length).fill(false));
  } else {
    // Add new manual visitor at mouse position
    const frequencies = [261.63, 329.63, 392.00, 440.00, 493.88, 523.25];
    const freqIndex = state.visitors.length % frequencies.length;
    
    state.manualVisitor = new Visitor(mouseX, mouseY, frequencies[freqIndex], MANUAL_VISITOR_ID);
    

    state.manualVisitor.volumeGate = new Tone.Gain(0.2);
    instruments.proximityPad.createVoice(SCALE_NOTES[freqIndex], state.manualVisitor.volumeGate);
    
    state.visitors.push(state.manualVisitor);
    visitorIdMap.set(MANUAL_VISITOR_ID, state.manualVisitor);
    console.log(`➕ Added manual visitor at (${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`);
    
    // Update lastSideMatrix size
    state.lastSideMatrix = Array(state.visitors.length).fill(null).map(() => Array(state.visitors.length).fill(false));
  }
}

// Update mouse position for manual visitor
function mouseMoved() {
  if (state.manualVisitor) {
    state.manualVisitor.setTargetPosition(mouseX, mouseY);
  }
}


  window.setup = setup;
  window.draw = draw;
  window.windowResized = windowResized;
  window.mousePressed = mousePressed;
  window.mouseMoved = mouseMoved;

  new p5(); // Start the p5 sketch
}