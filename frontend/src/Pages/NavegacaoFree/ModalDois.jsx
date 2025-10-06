import React from "react";
import "./ModalDois.css";

function ModalDois({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal" onClick={onClose}>
      <div
        className="modalGrande"
        style={{
          backgroundImage: "url('/Kepler-1649c.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="modalTitleOne">EXOPLANET</h2>
        <h3 className="modalTitle">Proxima Centauri b</h3>
        <p className="modal-body">The closest potentially habitable planet to Earth — our cosmic next-door neighbor.</p>
        <p className="modal-body">Just 4.24 light-years away, this world orbits Proxima Centauri, the nearest star to our Sun. It’s a rocky planet that may have temperatures where liquid water could exist — making it one of the most exciting finds in our galaxy.</p>
        <button className="modal-close" onClick={onClose}>Return to mission</button>
        <img 
          src="/ASTRONAUTA.png" 
          alt="Detalhe do planeta" 
          className="astronauta"/>
      </div>
    </div>
  );
}

export default ModalDois;
