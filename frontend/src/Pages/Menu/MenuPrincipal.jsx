import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './MenuPrincipal.css'

const Card = ({ title, tag, description, imageUrl, isHighlight = false, isSelected = false, onClick }) => (
  <div 
    className={`card ${isHighlight ? 'highlight' : ''} ${isSelected ? 'selected' : ''}`}
    onClick={onClick}
  >
    <div className="cardzinhoHeader">
      <h2 className="cardTitle">{title}</h2>

      <button
        className={`tag-button ${tag?.toLowerCase().replace(' ', '-')}`}
        onClick={(e) => {
          e.stopPropagation()
          alert(`${tag} plan selected!`)
        }}
      >
        {tag}
      </button>
    </div>

    <p className="cardDescricao">{description}</p>

    <div className="cardzinhoImage">
      <img src={imageUrl} alt={`Ilustração do plano ${title}`} className="card-image" />
    </div>
  </div>
)

function MenuPrincipal() {
  const navigate = useNavigate();
  const [selectedCard, setSelectedCard] = useState(null);

  const handleCardClick = (cardTitle, routePath) => {
    setSelectedCard(cardTitle);
    setTimeout(() => {
      navigate('../TelaAvancada')
    }, 300)
  }
    const handleStartExploring = () => {
      navigate('../TelaNavegacao')
    }

return (
  <div className="fundoContainer">
    <h1 className="tituloPrincipal">Choose your mission and explore the universe 🌌</h1>
    <div className="cardContainerzinho">
      <Card
        title="Free Explorer"
        tag="Basic"
        description="Fly your spaceship anywhere you want. Discover planets, land on them, and learn cool facts. Perfect for curious explorers!"
        imageUrl="./Rocket.png"
        isHighlight={true}
        isSelected={selectedCard === "Free Explorer"}
        onClick={() => handleStartExploring("Free Explorer", "/rota-free")}
      />

      <Card
        title="Planet Hunter"
        tag="Advanced"
        description="Ready for a challenge? Use our AI tester to search for planets. Upload a CSV data file, and the system will check if it really exists. Can you find the real ones?"
        imageUrl="./Planet.png"
        isHighlight={true}
        isSelected={selectedCard === "Planet Hunter"}
        onClick={() => handleCardClick("Planet Hunter", "/rota-hunter")}
      />
    </div>
       <h1 className="tituloPrincipal">You can always go back and try the other mode!</h1>
  </div>
  )
}

export default MenuPrincipal
