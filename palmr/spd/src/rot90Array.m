function OutArray = rot90Array(InArray)
OutArray = InArray;

for i = 1:3
    for j = 1:3
        OutArray(:,:,i,j) = fliplr( rot90(InArray(:,:,i,j)) );
    end
end