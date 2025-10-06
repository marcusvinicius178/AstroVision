import React from 'react'
import { useNavigate } from 'react-router-dom'
import './TelaInicial.css'

function TelaInicial() {

  const navigate = useNavigate()
  const handleStartExploring = () => {
    navigate('../MenuPrincipal')
  }

  return (
    <div className="tela-container">
      <video autoPlay loop muted className="video-fundo">
        <source src="/social_u57581.mp4" type="video/mp4" />
        Seu navegador não suporta vídeo.
      </video>

      <div className="conteudo">
        <div class="containerLogo">
          <img src="./Astrovision_1.svg" alt="Logo"/>
        </div>
        <p>
          You're about to start a journey through the stars! 🌟<br />
          Pilot your ship, search for new worlds, and learn amazing facts about exoplanets.
        </p>
          <button className="btn-explorar" onClick={handleStartExploring}>Start exploring</button>
      </div>
    </div>
  )
}

export default TelaInicial