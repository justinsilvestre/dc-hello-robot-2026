function getInitialCliffordState() {
  const state = {
    visitors: [],
    visitorIdMap: new Map(),

    xn: 0.1,
    yn: 0.1,
    a: 1.5,
    b: -1.8,
    c: 1.2,
    d: 0.9,
    /** Target value for d parameter */
    targetD: 0.9,
    /** How fast d transitions */
    dSmoothingSpeed: 0.02,
    t: 0,
    dt: 0.000001,

    resonanceSlices: [],
    densityThreshold: ITERATIONS_PER_FRAME * 0.05,

    // Tracking
    lastSideMatrix: null,
    screenCentroidX: 0,
    screenCentroidY: 0,

    // Manual visitor for testing
    manualVisitor: null,
    // Entropy tracking
    entropyGridResolution: 30, // doesn't seem to change
    entropyGridOccupancy: [],
    entropyOccupiedCount: 0,
    currentEntropy: 0,
    lastEntropyState: 0
  };


  //   // Start with no visitors - they will be added via OSC
  // // (or keep default visitors as fallback)
  // // visitors.push(new Visitor(width * 0.25, height * 0.25, 261.63, 'default-1'));
  // // visitors.push(new Visitor(width * 0.15, height * 0.50, 329.63, 'default-2'));
  // // visitors.push(new Visitor(width * 0.50, height * 0.75, 392.00, 'default-3'));
  // // visitors.push(new Visitor(width * 0.50, height * 0.25, 392.00, 'default-4'));
  
  // // Map default visitors
  // visitors.forEach(v => visitorIdMap.set(v.id, v));

  return state
}
