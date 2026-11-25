# examples (fixed rule: first validation row in each category)

## correct at rank 1 (label seen in train, rank 1)
- subject: Area of Simple Shapes
- question: James has answered a question on the area of a trapezium and got an answer of \( 54 \).

Behind the star he has written the units that he used.

\(
54 \, \bigstar 
\)

Which of the following units could be correct?
- correct answer: \( \mathrm{mm}^{2} \)
- student wrong answer: \( m \)
- true misconception: Does not know units of area should be squared
- predicted #1: Does not know units of area should be squared
- predicted #2: Believes squared units can be used to measure lengths
- predicted #3: Does not recognise a unit of area in word form

## correct at rank 2 or 3 (label seen in train, rank 3)
- subject: Area of Simple Shapes
- question: James has answered a question on the area of a trapezium and got an answer of \( 54 \).

Behind the star he has written the units that he used.

\(
54 \, \bigstar 
\)

Which of the following units could be correct?
- correct answer: \( \mathrm{mm}^{2} \)
- student wrong answer: \( \mathrm{km}^{3} \)
- true misconception: Does not know units of area should be squared
- predicted #1: Believes squared units can be used to measure lengths
- predicted #2: Does not know that units of volume are usually cubed
- predicted #3: Does not know units of area should be squared

## correct in top 25 but below 3 (label unseen in train, rank 9)
- subject: Trial and Improvement and Iterative Methods
- question: \(
x_{n+1}=5-\frac{1}{x_{n}}
\)

Using a starting value of \( x_{1}=2 \), what is the solution to \( x=5-\frac{1}{x} \) to \( 2 \) decimal places?
- correct answer: \( 4.79 \)
- student wrong answer: \( 4.78 \)
- true misconception: Stops iteration too early when solving equations
- predicted #1: Rounds to the wrong degree of accuracy (rounds too much)
- predicted #2: Believes substituting the starting value into an iteration formula immediately gives the solution
- predicted #3: Rounds to the wrong degree of accuracy (rounds too little)

## correct not in top 25 (label unseen in train, rank >25)
- subject: Real Life Graphs
- question: This graph shows how far Fido the dog is from his home.
What might the negative-sloping section represent? ![A graph with time (secs) on the horizontal axis and distance (m) on the vertical axis. The graph starts at the origin, travels in a straight line up an
- correct answer: Fido is walking home
- student wrong answer: Fido is accelerating
- true misconception: Believes the gradient of a distance-time graph is the acceleration
- predicted #1: Believes a downward slope on a speed-time graph shows positive acceleration
- predicted #2: Believes a downward slope on a distance-time graph means travelling away
- predicted #3: Does not interpret a sloping gradient as a change in rate
