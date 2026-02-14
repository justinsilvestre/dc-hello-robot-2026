class ResonanceSlice {
  constructor(note) {
    this.note = note;
    this.hitCount = 0;
  }

  resetHeat() {
    this.hitCount = 0;
  }

  registerHit() {
    this.hitCount++;
  }
}
