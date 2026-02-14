class Visitor {
  constructor(x, y, freq, id = null) {
    this.pos = createVector(x, y);
    this.targetPos = createVector(x, y); // Target position from OSC
    this.freq = freq;
    this.hitCount = 0;
    this.radius = VISITOR_RADIUS; // Use configurable radius
    this.synth = null;
    this.id = id || `visitor-${Math.random().toString(36).substr(2, 9)}`;
    this.handsRaised = false; // New property for hands status


    // Smoothing parameters - made more responsive
    this.smoothing = 0.25; // Increased from 0.15 for more responsiveness
    this.deadZone = 2; // Reduced from 5 for more sensitivity
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
      stroke(255, 40); // White when hands are down
    }
    ellipse(this.pos.x, this.pos.y, this.radius * 2);

    if (this.hitCount > 5) {
      let visualCappedHits = constrain(this.hitCount, 0, DENSITY_THRESHOLD);
      let opacity = map(visualCappedHits, 0, DENSITY_THRESHOLD, 0, 150);
      let pulseSize = map(visualCappedHits, 0, DENSITY_THRESHOLD, this.radius * 2, 120);

      // Different pulse colors based on hands status
      if (this.handsRaised) {
        fill(255, 255, 0, opacity); // Yellow pulse when hands raised
      } else {
        fill(255, opacity); // White pulse when hands down
      }
      noStroke();
      circle(this.pos.x, this.pos.y, pulseSize);
    }
  }
}
