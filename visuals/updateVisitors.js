/** Update visitors based on OSC data
 * Expected format: [id1, x1, y1, hands1, id2, x2, y2, hands2, ...]
 **/
function updateVisitors(state, args) {

  if (!args || args.length === 0) {
    console.log('📍 No people detected - keeping existing visitors');
    // When no people detected, slowly return to base value
    state.targetD = 0.9;
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
    const x = args[i + 1]; // x coordinate in meters
    const y = args[i + 2]; // y coordinate in meters
    const handsRaised = Boolean(args[i + 3]); // hands raised status

    if (handsRaised) {
      anyHandsRaised = true;
    }

    newVisitorIds.add(id);

    // Map floor coordinates to screen space
    // Floor coordinate system: (0,0) = bottom-left, (3,3.5) = top-right
    const screenX = map(x, 0, 3, 0, width);
    const screenY = map(y, 0, 3.5, height, 0); // Flip Y coordinate: larger Y values go toward top

    const handsStatus = handsRaised ? '🙌' : '👇';
    console.log(`📍 Person ${id}: floor(${x.toFixed(2)}, ${y.toFixed(2)}) -> screen(${screenX.toFixed(1)}, ${screenY.toFixed(1)}) hands: ${handsStatus}`);

    if (state.visitorIdMap.has(id)) {
      // Update existing visitor with smoothed position and hands data
      const visitor = state.visitorIdMap.get(id);
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
      const freqIndex = state.visitors.length % frequencies.length;
      const newVisitor = new Visitor(screenX, screenY, frequencies[freqIndex], id);
      newVisitor.handsRaised = handsRaised; // Set initial hands status
      newVisitor.volumeGate = new Tone.Gain(0.2);
      proximityPad.createVoice(SCALE_NOTES[freqIndex], newVisitor.volumeGate);

      state.visitors.push(newVisitor);
      state.visitorIdMap.set(id, newVisitor);
      console.log(`➕ Added visitor ${id} at floor coords (${x.toFixed(2)}, ${y.toFixed(2)}) -> screen (${screenX.toFixed(1)}, ${screenY.toFixed(1)}) hands: ${handsStatus}`);
    }
  }

  // Update target attractor parameter based on hands status (smoothly)
  state.targetD = anyHandsRaised ? 2.0 : 0.9;

  // Only log when target changes
  const prevTarget = state.d;
  if (Math.abs(state.targetD - prevTarget) > 0.01) {
    console.log(`🎯 Attractor target d set to: ${state.targetD} (hands raised: ${anyHandsRaised})`);
  }

  // Remove visitors that are no longer present
  const idsToRemove = [];
  state.visitorIdMap.forEach((visitor, id) => {
    if (!newVisitorIds.has(id) && !id.startsWith('default-')) {
      idsToRemove.push(id);
    }
  });

  idsToRemove.forEach(id => {
    const visitor = state.visitorIdMap.get(id);
    const index = state.visitors.indexOf(visitor);
    if (index > -1) {
      state.visitors.splice(index, 1);
      console.log(`➖ Removed visitor ${id}`);
    }
    state.visitorIdMap.delete(id);
  });

  // Update lastSideMatrix size if visitor count changed
  if (state.lastSideMatrix.length !== state.visitors.length) {
    state.lastSideMatrix = Array(state.visitors.length).fill(null).map(() => Array(state.visitors.length).fill(false));
  }
}
