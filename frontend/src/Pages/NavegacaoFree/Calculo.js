// Exemplo da lógica de planetas e distância
const PLANETS = [
  { id: 1, name: 'Kepler-164Bc', x: 200, y: 150 },
  // ... outros planetas
];
const PROXIMITY_RADIUS = 80; // Distância máxima para "clicar"

// No componente SpaceExplorer:
const checkProximity = (rocketX, rocketY) => {
  for (const planet of PLANETS) {
    const distance = Math.sqrt(
      Math.pow(planet.x - rocketX, 2) + Math.pow(planet.y - rocketY, 2)
    );
    if (distance < PROXIMITY_RADIUS) {
      return planet; // Retorna o planeta que está perto
    }
  }
  return null;
};

// ...
const closePlanet = checkProximity(rocketPos.x, rocketPos.y);
// Armazene closePlanet no estado e use-o para renderizar o tooltip