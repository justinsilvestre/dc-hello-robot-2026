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

  updateSound(columnIndex, resonanceSlices) {
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
