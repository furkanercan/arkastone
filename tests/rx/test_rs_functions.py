# import pytest
# import random
# import galois
# from src.utils import galois_field
# from src.tx.encoders.encoder_rs import ReedSolomonEncoder
# from src.rx.decoders.rs.dec_rs import ReedSolomonDecoder

# def test_rs_syndrome_computation():
#     rs_decoder = ReedSolomonDecoder(n=255, k=223, m=8, primitive_poly=0x11D)
#     # Simulate a received codeword with no errors (should have zero syndromes)
#     received_no_error = [0] * 255  # All-zero codeword (valid codeword)
#     syndromes_no_error = rs_decoder.compute_syndromes(received_no_error)
#     assert all(s == 0 for s in syndromes_no_error), f"Expected zero syndromes, got {syndromes_no_error}"
    
#     # Simulate a received codeword with an error (introduce error at position 10)
#     received_with_error = [0] * 255
#     received_with_error[10] = 5  # Inject an error
#     syndromes_with_error = rs_decoder.compute_syndromes(received_with_error)
#     assert any(s != 0 for s in syndromes_with_error), f"Expected nonzero syndromes, got {syndromes_with_error}"
    
#     print("Test passed: Syndrome computation is working correctly.")

# def test_berlekamp_massey_no_errors():
#     rs_decoder = ReedSolomonDecoder(n=255, k=223, m=8, primitive_poly=0x11D)
#     syndromes = [0] * 32  # No errors → all syndromes are zero
#     locator_poly = rs_decoder.berlekamp_massey(syndromes)
#     assert locator_poly == [1], f"Expected [1], got {locator_poly}"

# def test_berlekamp_massey_toy_example():
#     GF = galois.GF(7)

#     # Let's create a syndrome vector from a toy example with one error
#     # Suppose the true locator poly is Λ(x) = 1 + 2x
#     # Generate syndrome sequence s_n = Λ₀·sₙ + Λ₁·sₙ₋₁ = 0 → recurrence
#     # We'll use a known example
#     syndromes = GF([1, 3, 0, 5, 1, 6])  # Picked to have degree-1 locator

#     decoder = ReedSolomonDecoder(n=6, k=4)  # Dummy values, not used here

#     custom_poly = decoder.berlekamp_massey(syndromes)
#     builtin_poly = decoder.berlekamp_massey_builtin(syndromes)

#     print(f"Custom locator poly:  {custom_poly}")
#     print(f"Builtin locator poly: {builtin_poly}")

#     # Normalize both to monic
#     def normalize(poly):
#         poly = list(poly)
#         return [c / poly[0] for c in poly] if poly[0] != 1 else poly

#     assert normalize(custom_poly) == normalize(builtin_poly), "Polynomials do not match!"


# def test_rs_1error_over_gf8():
#     # Setup GF(2^3) with primitive polynomial x^3 + x + 1 = 0xB
#     primitive_poly = 0xB
#     GF = galois_field.GF(3, primitive_poly)
#     n, k = 7, 5

#     # Create encoder and decoder using your implementation
#     encoder = ReedSolomonEncoder(n=n, k=k, m=3, primitive_poly=primitive_poly)
#     decoder = ReedSolomonDecoder(n=n, k=k, m=3, primitive_poly=primitive_poly)

#     # Generate a random message of length k
#     # message = [int(x) for x in GF.Random(k)]

#     message = [random.randint(0, n - 1) for _ in range(k)]
#     codeword = encoder.encode(message)

#     # Inject 1 error at a known location
#     error_pos = 3
#     corrupted = codeword.copy()
#     corrupted[error_pos] ^= 0b001  # Add a small error

#     # Compute syndromes using your decoder
#     syndromes = decoder.compute_syndromes(corrupted)

#     # Run custom Berlekamp-Massey
#     locator_custom = decoder.berlekamp_massey(syndromes)

#     # Run Galois built-in BM for comparison
#     locator_builtin = galois.berlekamp_massey(syndromes_gf).coeffs.tolist()

#     def normalize(poly, field):
#         # Cast to int first, then wrap into field
#         # Handles float like -1.0, 1.0 and maps correctly to GF(2^m)
#         poly = [field(int(round(x))) for x in poly]
#         return [x / poly[0] for x in poly] if poly[0] != 1 else poly


#     locator_custom_norm = normalize(locator_custom, GF)
#     locator_builtin_norm = normalize(locator_builtin, GF)

#     # Print info
#     print(f"Message:            {message}")
#     print(f"Original codeword:  {codeword}")
#     print(f"Corrupted codeword: {corrupted}")
#     print(f"Syndromes:          {syndromes}")
#     print(f"Custom λ(x):        {galois.Poly(locator_custom, field=GF)}")
#     print(f"Builtin λ(x):       {galois.Poly(locator_builtin, field=GF)}")
#     print(f"Custom λ(x):        {galois.Poly(locator_custom_norm, field=GF)}")
#     print(f"Builtin λ(x):       {galois.Poly(locator_builtin_norm, field=GF)}")

#     assert locator_custom_norm == locator_builtin_norm, "Mismatch in locator polynomial!"



# def test_berlekamp_massey_single_error():
#     """Test Berlekamp-Massey algorithm by simulating an RS-coded message with a single error."""
#     n, k = 255, 223  # RS(255,223) standard
#     GF = galois.GF(2**8)
    
#     encoder = ReedSolomonEncoder(n=n, k=k)
#     decoder = ReedSolomonDecoder(n=n, k=k)

#     # Step 1: Generate a random message (length k)
#     message = [random.randint(0, 255) for _ in range(encoder.k)]

#     # Step 2: Encode the message
#     codeword = encoder.encode(message)

#     # Step 3: Introduce a single error in a random position
#     error_position = random.randint(0, encoder.n - 1)
#     original_value = codeword[error_position]
#     codeword[error_position] ^= 0xFF  # Flip some bits arbitrarily

#     print(f"Injected error at position {error_position}: {original_value} → {codeword[error_position]}")

#     # Step 4: Compute syndromes
#     syndromes = decoder.compute_syndromes(codeword)
#     syndromes_gf = GF(syndromes)

#     # Step 5: Run custom Berlekamp-Massey
#     # locator_poly_custom = decoder.berlekamp_massey(syndromes)
#     locator_poly_custom = decoder.berlekamp_massey(syndromes[:2])
#     locator_poly_galois = galois.berlekamp_massey(GF(syndromes[:2])).coeffs.tolist()

#     # Normalize both to monic form for comparison
#     def normalize(poly):
#         poly = list(poly)  # In case it's a galois FieldArray
#         return [c / poly[0] for c in poly] if poly[0] != 1 else poly

#     # Step 6: Run Galois built-in Berlekamp-Massey
#     locator_poly_galois = galois.berlekamp_massey(syndromes_gf).coeffs  # returns FieldArray
#     locator_poly_galois_list = locator_poly_galois.tolist()  # convert to Python list

#     # Step 7: Compare results
#     print(f"Custom  error locator polynomial: {locator_poly_custom}")
#     print(f"Galois  error locator polynomial: {locator_poly_galois_list}")

#     assert normalize(locator_poly_custom) == normalize(locator_poly_galois_list), "Mismatch between custom and galois BM results"


#     print("✅ Berlekamp-Massey match confirmed.")

# def test_berlekamp_massey_single_error():
#     rs_decoder = ReedSolomonDecoder(n=255, k=223)
#     syndromes = [1, 2, 4, 8, 16, 32] + [0] * 26  # Simulated error syndrome
#     locator_poly = rs_decoder.berlekamp_massey(syndromes)
#     assert len(locator_poly) == 2, f"Expected degree-1 polynomial, got {locator_poly}"
#     assert locator_poly[0] == 1, "First term should be 1 (monic polynomial)"

# def test_berlekamp_massey_two_errors():
#     rs_decoder = ReedSolomonDecoder(n=255, k=223)
#     syndromes = [5, 10, 20, 40, 80, 160] + [0] * 26  # Simulated double error
#     locator_poly = rs_decoder.berlekamp_massey(syndromes)
#     assert len(locator_poly) == 3, f"Expected degree-2 polynomial, got {locator_poly}"
#     assert locator_poly[0] == 1, "First term should be 1 (monic polynomial)"
