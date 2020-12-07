import SedmaDatatypes

instance Eq Rank where
    R7 == R7 = True
    R8 == R8 = True
    R9 == R9 = True
    R10 == R10 = True
    RJ == RJ = True
    RQ == RQ = True
    RK == RK = True
    RA == RA = True
    _ == _ = False

pointsOf :: Rank -> Int
pointsOf R10 = 10
pointsOf RA = 10
pointsOf _ = 0

-- are the player rank and first card rank same
isTrickRank :: Rank -> Maybe Rank -> Bool
isTrickRank _ Nothing = False
isTrickRank R7 _ = True
isTrickRank rank (Just trickRank) = rank == trickRank

-- round :: 4 cards -> who is playing now -> current winner -> current score -> first card rank -> (AC won, AC score, BD won, BD score)
roundResult :: Cards -> Team -> Team -> Int -> Maybe Rank -> (Bool, Int, Bool, Int)
roundResult [] _ AC score _ = (True, score, False, 0)
roundResult [] _ BD score _ = (False, 0, True, score)
roundResult ((Card _ rank):xs) team winner score trickRank = roundResult xs nextTeam newWinner newScore newTrickRank
    where
        nextTeam = if (team == AC) then BD else AC
        newWinner = if(isTrickRank rank trickRank) then team else winner
        newScore = score + (pointsOf rank)
        newTrickRank = if (trickRank == Nothing) then (Just rank) else trickRank

-- stock -> leader -> (AC won trick(s), AC score, BD won trick(s), BD score)
allRoundsResult :: Cards -> Team -> (Bool, Int, Bool, Int) -> (Bool, Int, Bool, Int)
allRoundsResult [] _ result = result
allRoundsResult cards curTeam (hasTrickAC, scoreAC, hasTrickBD, scoreBD) = allRoundsResult remainingCards newTeam (trAC, scAC, trBD, scBD) 
    where
        remainingCards = drop 4 cards
        (isRoundWinnerAC, pointsAC, isRoundWinnerBD, pointsBD) = roundResult (take 4 cards) curTeam curTeam 0 Nothing
        newTeam = if (isRoundWinnerAC) then AC else BD
        lastTrickPoints = if (null remainingCards) then 1 else 0
        -- update winner flags
        trAC = or [isRoundWinnerAC, hasTrickAC] 
        trBD = or [isRoundWinnerBD, hasTrickBD]
        -- new score
        scAC = scoreAC + pointsAC + if (isRoundWinnerAC) then lastTrickPoints else 0 
        scBD = scoreBD + pointsBD + if (isRoundWinnerBD) then lastTrickPoints else 0

replay :: Cards -> Maybe Winner
replay cards
    | (length cards) /= 32 = Nothing
    | not hasTrickBD = Just(AC, Three)
    | not hasTrickAC = Just(BD, Three)
    | scoreBD == 0 = Just(AC, Two)
    | scoreAC == 0 = Just(BD, Two)
    | scoreAC > scoreBD = Just(AC, One)
    | scoreAC < scoreBD = Just(BD, One)
    where
        (hasTrickAC, scoreAC, hasTrickBD, scoreBD) = allRoundsResult cards AC (False, 0, False, 0)