function R = Rotation(X,Y)
% Find a rotation matrix that minimizes
% \|X - Y*R\|_F

if any(size(X) ~= size(Y))
    error('X and Y must have the same size.');
end

[U, ~, V] = svd(X'*Y);
R = V * U';

end