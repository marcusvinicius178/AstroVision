import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import TelaInicial from './Pages/TelaInicial/TelaInicial'
import MenuPrincipal from './Pages/Menu/MenuPrincipal'
import TelaNavegacao from './Pages/NavegacaoFree/TelaNavegacao'
import TelaAvancada from './Pages/PesquisaAvancada/TelaAvancada'

function App() {
  return (
    <BrowserRouter>
    <Routes>
      <Route path="/" element={<TelaInicial />} />
      <Route path="/MenuPrincipal" element={<MenuPrincipal />} />
      <Route path="/TelaNavegacao" element={<TelaNavegacao />} />
      <Route path="/TelaAvancada" element={<TelaAvancada />} />
    </Routes>
    </BrowserRouter>
  )
}

export default App
