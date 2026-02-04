# 🎬 Script de Démonstration Marketplace Incognito (120s)

## 🎯 Objectif
Démontrer l'écosystème complet de la marketplace confidentielle : du dépôt privé jusqu'à la messagerie chiffrée, en passant par l'achat et la gestion d'escrow.

---

## ⏱️ Chronologie Détaillée (2 minutes)

### **[0:00 - 0:20] Landing Page → Onboarding (20s)**

**Action** : Ouvrir `http://localhost:8080`

**Points clés à montrer** :
1. **Landing Page** (5s)
   - Tagline : "Privacy-First Decentralized Marketplace"
   - Bouton "Get Started" bien visible
   
2. **Onboarding** (15s)
   - Cliquer sur "Get Started"
   - **Import de wallet** : Sélectionner un keypair existant (ex: `keys/userA.json`)
   - Validation automatique → Redirection vers Dashboard

**Script oral** :
> "Bienvenue sur Incognito Protocol. L'onboarding est simple : un seul clic pour importer votre wallet Solana existant, et vous êtes prêt à utiliser la marketplace de manière totalement confidentielle."

---

### **[0:20 - 0:45] Dashboard - Vue d'ensemble (25s)**

**URL** : `http://localhost:8080/app` (page par défaut)

**Points clés à montrer** :
1. **Header** (5s)
   - Wallet address raccourcie (ex: `7Rj...H7S`)
   - Badge de connexion (vert = connecté)
   
2. **Metrics Cards** (10s)
   - **Total Balance** : Affiche le SOL disponible
   - **Privacy Notes** : Nombre de notes chiffrées disponibles
   - **Active Orders** : Commandes en cours
   - **Escrow Balance** : Fonds en garantie

3. **Quick Actions** (10s)
   - Bouton "Create Note" (conversion SOL → Note privée)
   - Bouton "Browse Marketplace"
   - Bouton "Check Messages"

**Script oral** :
> "Le Dashboard centralise tout : vos soldes publics et privés, vos commandes en escrow, et vos actions rapides. Remarquez que les 'Privacy Notes' sont vos tokens confidentiels issus du pool anonyme."

---

### **[0:45 - 1:10] Marketplace - Achat Privé (25s)**

**URL** : `http://localhost:8080/app/marketplace`

**Scénario** : Acheter un produit avec une Privacy Note

**Points clés à montrer** :
1. **Liste des produits** (8s)
   - Catalogue avec images, titres, prix
   - Badge "Confidential Payment" sur chaque item
   - Filtres par catégorie (Electronics, Fashion, etc.)

2. **Détail produit** (7s)
   - Clic sur un produit (ex: "Wireless Headphones - 0.5 SOL")
   - Modal avec description complète
   - Sélecteur de quantité

3. **Checkout confidentiel** (10s)
   - Bouton "Buy with Privacy Note"
   - **Sélection de la note** : Dropdown montrant vos notes disponibles
   - Clic "Confirm Purchase"
   - Toast de succès : "Order placed! Escrow created."

**Script oral** :
> "Lors de l'achat, je choisis une Privacy Note au lieu de payer directement en SOL. C'est ici que la magie opère : le vendeur ne voit pas d'où viennent les fonds grâce au pool de confidentialité. L'escrow est créé automatiquement."

---

### **[1:10 - 1:35] Orders & Escrow - Gestion sécurisée (25s)**

**URL** : `http://localhost:8080/app/orders`

**Points clés à montrer** :
1. **Onglets Buyer/Seller** (5s)
   - Basculer entre "My Purchases" et "My Sales"

2. **Détails d'une commande** (12s)
   - Statut : `PENDING` → Badge orange
   - Timeline :
     - ✓ Order Placed
     - 🔄 Waiting for Shipment
     - ⏳ Awaiting Delivery
   - **Bouton vendeur** : "Mark as Shipped" → Input tracking number
   - **Bouton acheteur** : "Release Funds" (après livraison)

3. **Actions Escrow** (8s)
   - Simuler "Mark as Shipped" (vendeur)
   - Toast : "Tracking number updated"
   - Montrer bouton "Release Funds" (acheteur)

**Script oral** :
> "L'escrow protège les deux parties. Le vendeur expédie et ajoute le numéro de suivi. L'acheteur, une fois le produit reçu, libère les fonds en un clic. En cas de litige, un bouton 'Dispute' est disponible."

---

### **[1:35 - 2:00] Messages - Inbox Chiffrée (25s)**

**URL** : `http://localhost:8080/app/messages`

**Points clés à montrer** :
1. **Liste des conversations** (8s)
   - Messages chiffrés de bout en bout
   - Preview : "New order #abc123..." (chiffré côté serveur)
   - Badge "Unread" (nombre de messages non lus)

2. **Lecture d'un message** (10s)
   - Clic sur une conversation
   - Déchiffrement local → Affichage du contenu
   - **Exemple** : "Your order has been shipped! Tracking: 1Z999..."

3. **Envoi de réponse** (7s)
   - Taper "Thanks! Looking forward to receiving it."
   - Clic "Send" → Chiffrement automatique
   - Toast : "Message sent (encrypted)"

**Script oral** :
> "La messagerie est entièrement E2EE. Le serveur stocke des blobs chiffrés, seuls le vendeur et l'acheteur peuvent lire le contenu. Parfait pour échanger des informations de livraison sans compromettre la confidentialité."

---

## 🎨 Bonus : Points Visuels à Souligner

### **Design & UX**
- **Glassmorphism** : Effets de flou sur les cards (moderne)
- **Dark Mode** : Palette sombre avec accents violets/bleus
- **Animations** : Transitions fluides entre les pages
- **Responsive** : Montrer un resize de fenêtre (mobile-friendly)

### **Badges de Statut**
- 🟢 `PENDING` → Orange
- 🟢 `SHIPPED` → Bleu
- 🟢 `RELEASED` → Vert
- 🔴 `DISPUTED` → Rouge

---

## 📋 Checklist Pré-Démo

### Avant de commencer :
- [ ] **API lancée** : `uvicorn services.api.app:app --host 0.0.0.0 --port 8001`
- [ ] **Frontend lancé** : `cd web-interface && npm run dev`
- [ ] **Wallet importé** : Avoir un wallet avec SOL et notes disponibles
- [ ] **Données de test** :
  - Au moins 1 Privacy Note (créer via Dashboard si besoin)
  - Au moins 1 listing dans la marketplace
  - Au moins 1 message dans l'inbox

### Commandes Rapides :
```bash
# Terminal 1 : API
cd /Users/alex/Desktop/incognito-protocol-1
uvicorn services.api.app:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 : Frontend
cd /Users/alex/Desktop/incognito-protocol-1/web-interface
npm run dev

# Terminal 3 : Arcium (si besoin de MPC)
cd /Users/alex/Desktop/incognito-protocol-1/contracts/incognito
arcium localnet
```

---

## 🎯 Messages Clés à Transmettre

1. **Confidentialité Native** : Les paiements via Privacy Notes cassent le graph de transactions
2. **Sécurité Escrow** : Protection automatique vendeur/acheteur sans tiers de confiance
3. **E2EE Messaging** : Communication ultra-sécurisée pour les détails de livraison
4. **UX Moderne** : Interface intuitive malgré la complexité cryptographique sous-jacente

---

## 🔄 Variante Alternative (si plus de temps)

### **[BONUS +30s] : Créer une Privacy Note en live**

**URL** : `http://localhost:8080/app/notes`

1. Clic "Create New Note"
2. Input : `0.5 SOL`
3. Bouton "Deposit to Pool"
4. ⏳ Attente confirmation (5-10s)
5. ✅ Note apparaît dans la liste avec commitment hash

**Script** :
> "Créer une Note, c'est déposer des SOL dans le pool anonyme. Une fois inside, impossible de tracer d'où viennent ces fonds. C'est le cœur du système de confidentialité."

---

## 🎬 Conclusion (dernières 5 secondes)

**Slide final ou écran récap** :
- Logo Incognito Protocol
- Tagline : "Privacy is a Right, Not a Feature"
- GitHub : `github.com/ChupaSOLANA/incognito-protocol`

**Script de clôture** :
> "Incognito Protocol prouve qu'on peut avoir une marketplace décentralisée ET confidentielle, sans compromis sur l'UX. Merci !"
