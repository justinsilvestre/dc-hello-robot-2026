/**
 * When set to `true`,
 * press "I" to add a simulated person and "O" to remove one.
 */
let simulate = true;

// Simulation state
const simState = {
  people: [],
  nextId: 1,
  roomWidth: 3.0,    // room is 3 x 3.5 units
  roomHeight: 3.5,
  updateRate: 200,   // ~5 times per second
  intervalId: null
};



function listenForOscMessages(onPositionsMessage) {
  if (simulate) {
    console.log('🤖 Starting position simulation...');
    console.log('   Press "I" to add a person, "O" to remove a person');
    
    // Start with 1 person
    simState.people.push(new SimulatedVisitor(simState.nextId++));
    
    // Set up keyboard listeners
    document.addEventListener('keydown', (event) => {
      if (event.key.toLowerCase() === 'i') {
        // Add new person
        simState.people.push(new SimulatedVisitor(simState.nextId++));
        console.log(`👋 Person ${simState.nextId - 1} entered (total: ${simState.people.length})`);
      } else if (event.key.toLowerCase() === 'o') {
        // Remove a person
        if (simState.people.length > 0) {
          const removed = simState.people.pop();
          console.log(`🚪 Person ${removed.id} left (remaining: ${simState.people.length})`);
        }
      }
    });
    
    // Start simulation loop
    simState.intervalId = setInterval(() => {
      // Update all people
      simState.people.forEach(person => person.update());
      
      // Create position message in expected format:
      // [id1, x1, y1, handsUp1, id2, x2, y2, handsUp2, ...]
      const positionData = [];
      simState.people.forEach(person => {
        positionData.push(
          person.id,
          person.x,
          person.y,
          person.handsRaised ? 1 : 0
        );
      });
      
      
      if (positionData) {
        onPositionsMessage(positionData);
      }
    }, simState.updateRate);
    
    return
  }
  // Connect to OSC bridge server
  console.log('Attempting to connect to Socket.IO server...');
  const socket = io('http://localhost:8080');
  
  socket.on('connect', () => {
    console.log('✅ Connected to OSC bridge');
  });
  
  socket.on('connect_error', (error) => {
    console.error('❌ Connection error:', error);
  });
  
  // Listen for people positions from skeleton2.py
  // Expected format: [id1, x1, y1, handUp1, id2, x2, y2, handsUp2, ...]
  socket.on('/people/positions', (args) => {
    // console.log('👥 People positions received:', args);
    onPositionsMessage(args);
  });
  
  // Listen for ANY OSC message to debug
  socket.onAny((eventName, ...args) => {
    // console.log('🔔 OSC event:', eventName, args);
  });
  
  socket.on('disconnect', () => {
    console.log('❌ Disconnected from OSC bridge');
  });
}



class SimulatedVisitor {
  constructor(id) {
    this.id = id;
    // Start at random position
    this.x = Math.random() * simState.roomWidth;
    this.y = Math.random() * simState.roomHeight;
    this.handsRaised = false;
    
    // Movement parameters
    this.vx = (Math.random() - 0.5) * 0.002; // slow movement
    this.vy = (Math.random() - 0.5) * 0.002;
    this.targetX = this.x;
    this.targetY = this.y;
    
    // Change direction/target every few seconds
    this.directionChangeCounter = 0;
    this.directionChangeInterval = Math.floor(3000 / simState.updateRate); // ~3 seconds
  }
  
  update() {
    // Occasionally change direction/target
    this.directionChangeCounter++;
    if (this.directionChangeCounter >= this.directionChangeInterval) {
      this.directionChangeCounter = 0;
      this.targetX = Math.random() * simState.roomWidth;
      this.targetY = Math.random() * simState.roomHeight;
      this.directionChangeInterval = Math.floor((2 + Math.random() * 4) * 1000 / simState.updateRate); // 2-6 seconds
      
      // Occasionally raise hands
      this.handsRaised = Math.random() < 0.1; // 10% chance
    }
    
    // Move towards target slowly
    const dx = this.targetX - this.x;
    const dy = this.targetY - this.y;
    this.x += dx * 0.01; // very slow interpolation
    this.y += dy * 0.01;
    
  console.log(this.id, this.x, this.y)

    // Keep within bounds
    this.x = Math.max(0, Math.min(simState.roomWidth, this.x));
    this.y = Math.max(0, Math.min(simState.roomHeight, this.y));
  }
}