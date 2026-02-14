function listenForOscMessages(onPositionsMessage) {
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
  // Expected format: [id1, x1, y1, id2, x2, y2, ...]
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
